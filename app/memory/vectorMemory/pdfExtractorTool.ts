import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import type { Document } from '@langchain/core/documents';
import axios from 'axios';
import fs from 'node:fs';
import path from 'node:path';
import type { ITool } from '../../agents/agents';

export type PdfType = 'text-based' | 'image-based';

export interface ExtractedPage {
  pageNumber: number;
  text: string;
  characterCount: number;
}

export interface PdfExtractionResult {
  success: boolean;
  source: string;
  pdfType: PdfType;
  totalPages: number;
  totalCharacters: number;
  content: string;
  pages: ExtractedPage[];
  metadata?: Record<string, any>;
  message?: string;
  error?: string;
}

export class PdfExtractor {
  // Configurable threshold: average characters per page below which a PDF is considered scanned/image-based
  private scannedThresholdCharsPerPage: number;

  constructor(options?: { scannedThresholdCharsPerPage?: number }) {
    this.scannedThresholdCharsPerPage = options?.scannedThresholdCharsPerPage ?? 40;
  }

  /**
   * Validates if the source is supported (must be a valid URL or existing local file with .pdf extension)
   */
  public validateSource(source: string): { isValid: boolean; isUrl: boolean; error?: string } {
    if (!source || typeof source !== 'string') {
      return { isValid: false, isUrl: false, error: 'Source must be a non-empty string path or URL.' };
    }

    const trimmed = source.trim();
    const isUrl = /^https?:\/\//i.test(trimmed);

    if (isUrl) {
      try {
        const parsed = new URL(trimmed);
        const pathname = parsed.pathname.toLowerCase();
        if (!pathname.endsWith('.pdf') && !parsed.search.toLowerCase().includes('.pdf')) {
          console.warn(`[PdfExtractor] Warning: URL does not explicitly end with .pdf: ${trimmed}`);
        }
        return { isValid: true, isUrl: true };
      } catch (err: any) {
        return { isValid: false, isUrl: true, error: `Invalid URL format: ${err.message}` };
      }
    }

    // Local file path validation
    const ext = path.extname(trimmed).toLowerCase();
    if (ext !== '.pdf') {
      return {
        isValid: false,
        isUrl: false,
        error: `Unsupported file extension "${ext}". Only .pdf files are supported.`
      };
    }

    if (!fs.existsSync(trimmed)) {
      return {
        isValid: false,
        isUrl: false,
        error: `Local PDF file not found at: ${trimmed}`
      };
    }

    return { isValid: true, isUrl: false };
  }

  /**
   * Loads PDF documents using LangChain's PDFLoader.
   * Handles both local file paths and remote URLs.
   */
  public async loadPdfDocuments(source: string): Promise<{ docs: Document[]; isUrl: boolean }> {
    const validation = this.validateSource(source);
    if (!validation.isValid) {
      throw new Error(validation.error);
    }

    const trimmed = source.trim();

    if (validation.isUrl) {
      // Download remote PDF as ArrayBuffer and pass Blob to LangChain PDFLoader
      const response = await axios.get(trimmed, {
        responseType: 'arraybuffer',
        headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' }
      });

      const blob = new Blob([response.data], { type: 'application/pdf' });
      const loader = new PDFLoader(blob, { splitPages: true });
      const docs = await loader.load();
      return { docs, isUrl: true };
    } else {
      // Local file
      const loader = new PDFLoader(trimmed, { splitPages: true });
      const docs = await loader.load();
      return { docs, isUrl: false };
    }
  }

  /**
   * Detects whether the loaded PDF is text-based (has digital text streams)
   * or image-based (scanned document / raster images with little or no extractable text).
   */
  public detectPdfType(docs: Document[]): {
    pdfType: PdfType;
    totalPages: number;
    totalCharacters: number;
    avgCharsPerPage: number;
  } {
    const totalPages = docs.length;
    if (totalPages === 0) {
      return { pdfType: 'image-based', totalPages: 0, totalCharacters: 0, avgCharsPerPage: 0 };
    }

    const totalCharacters = docs.reduce(
      (acc, doc) => acc + (doc.pageContent?.trim().length || 0),
      0
    );
    const avgCharsPerPage = totalCharacters / totalPages;

    const pdfType: PdfType =
      avgCharsPerPage < this.scannedThresholdCharsPerPage ? 'image-based' : 'text-based';

    return {
      pdfType,
      totalPages,
      totalCharacters,
      avgCharsPerPage
    };
  }

  /**
   * Extracts content from a digital, text-based PDF.
   */
  public async extractTextBased(docs: Document[], source: string): Promise<PdfExtractionResult> {
    const pages: ExtractedPage[] = docs.map((doc, index) => {
      const pageText = doc.pageContent || '';
      return {
        pageNumber: (doc.metadata?.loc?.pageNumber as number) || index + 1,
        text: pageText.trim(),
        characterCount: pageText.trim().length
      };
    });

    const combinedContent = pages
      .map((p) => `--- Page ${p.pageNumber} ---\n${p.text}`)
      .join('\n\n');

    const totalCharacters = pages.reduce((acc, p) => acc + p.characterCount, 0);

    return {
      success: true,
      source,
      pdfType: 'text-based',
      totalPages: docs.length,
      totalCharacters,
      content: combinedContent,
      pages,
      metadata: docs[0]?.metadata || {}
    };
  }

  /**
   * Handles image-based / scanned PDFs.
   * Scanned PDFs contain minimal digital text and require multimodal vision or OCR processing.
   */
  public async extractImageBased(docs: Document[], source: string): Promise<PdfExtractionResult> {
    const pages: ExtractedPage[] = docs.map((doc, index) => ({
      pageNumber: (doc.metadata?.loc?.pageNumber as number) || index + 1,
      text: (doc.pageContent || '').trim(),
      characterCount: (doc.pageContent || '').trim().length
    }));

    const totalCharacters = pages.reduce((acc, p) => acc + p.characterCount, 0);

    return {
      success: true,
      source,
      pdfType: 'image-based',
      totalPages: docs.length,
      totalCharacters,
      content: pages
        .map((p) => `--- Page ${p.pageNumber} [Scanned/Image] ---\n${p.text}`)
        .join('\n\n'),
      pages,
      message:
        'This PDF appears to be scanned or image-based with minimal embedded text. Multimodal OCR or Vision processing is recommended for full text recovery.'
    };
  }

  /**
   * Unified extraction pipeline:
   * 1. Validates the source (URL or local path and .pdf extension)
   * 2. Loads via LangChain PDFLoader
   * 3. Detects if text-based or image-based
   * 4. Dispatches to the appropriate extraction method
   */
  public async extract(source: string): Promise<PdfExtractionResult> {
    try {
      const { docs } = await this.loadPdfDocuments(source);
      const detection = this.detectPdfType(docs);

      if (detection.pdfType === 'text-based') {
        return await this.extractTextBased(docs, source);
      } else {
        return await this.extractImageBased(docs, source);
      }
    } catch (err: any) {
      return {
        success: false,
        source,
        pdfType: 'text-based',
        totalPages: 0,
        totalCharacters: 0,
        content: '',
        pages: [],
        error: err?.message || String(err)
      };
    }
  }

  // Helper to parse tool inputs (JSON object or raw string)
  private parseToolInput(input: string): Record<string, any> {
    try {
      const parsed = JSON.parse(input);
      if (typeof parsed === 'object' && parsed !== null) return parsed;
      return { source: String(parsed) };
    } catch {
      return { source: input.trim() };
    }
  }

  /**
   * Exposes ITool definitions matching the memory and CLI tool pattern,
   * ready to be consumed by VectorMaster.
   */
  public getItools(): ITool[] {
    return [
      {
        name: 'extractPdf',
        description:
          'Loads a PDF from a local file path or URL, validates it, detects whether it is text-based or scanned (image-based), and extracts its page content.',
        doc: 'extractPdf({"source": "https://example.com/document.pdf" | "C:/path/to/doc.pdf"})',
        executor: async (input: string): Promise<string> => {
          try {
            const data = this.parseToolInput(input);
            const source = data.source || data.url || data.path || data.query;
            if (!source) {
              return JSON.stringify({ error: 'Missing required field: "source" (URL or file path)' });
            }
            const result = await this.extract(source);
            return JSON.stringify(result);
          } catch (err: any) {
            return JSON.stringify({ error: err?.message || String(err) });
          }
        }
      },
      {
        name: 'detectPdfType',
        description:
          'Inspects a PDF from a URL or local file path to check if it is text-based (digital) or image-based (scanned).',
        doc: 'detectPdfType({"source": "https://example.com/document.pdf"})',
        executor: async (input: string): Promise<string> => {
          try {
            const data = this.parseToolInput(input);
            const source = data.source || data.url || data.path || data.query;
            if (!source) {
              return JSON.stringify({ error: 'Missing required field: "source"' });
            }
            const { docs } = await this.loadPdfDocuments(source);
            const detection = this.detectPdfType(docs);
            return JSON.stringify({ success: true, source, ...detection });
          } catch (err: any) {
            return JSON.stringify({ error: err?.message || String(err) });
          }
        }
      }
    ];
  }
}

// Standalone default instance
export const pdfExtractor = new PdfExtractor();
