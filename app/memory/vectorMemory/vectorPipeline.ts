import OpenAI from 'openai';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { getPineconeIndex } from './pineconeClient';
import { pdfExtractor, PdfExtractor, type PdfExtractionResult } from './pdfExtractorTool';
import {
  audioExtractor,
  AudioExtractor,
  formatTimestamp,
  type AudioExtractionResult
} from './audioExtractorTool';
import type { ITool } from '../../agents/agents';
import crypto from 'node:crypto';
import 'dotenv/config';

export interface ChunkWithMetadata {
  id: string;
  text: string;
  pageNumber?: number;
  startTimeSeconds?: number;
  endTimeSeconds?: number;
  timeLabel?: string;
  chunkIndex: number;
  source: string;
  mediaType?: string;
  metadata?: Record<string, any>;
}

export interface IngestionResult {
  success: boolean;
  source: string;
  pdfType: string;
  totalPages: number;
  totalChunks: number;
  vectorsUploaded: number;
  namespace?: string;
  message?: string;
  error?: string;
}

export interface AudioIngestionResult {
  success: boolean;
  source: string;
  mediaType: string;
  durationSeconds: number;
  totalSegments: number;
  totalChunks: number;
  vectorsUploaded: number;
  detectedLanguage?: string;
  jobId?: string;
  namespace?: string;
  message?: string;
  error?: string;
}

export interface VectorSearchResult {
  id: string;
  score: number;
  text: string;
  source: string;
  pageNumber?: number;
  startTimeSeconds?: number;
  endTimeSeconds?: number;
  timeLabel?: string;
  mediaType?: string;
  metadata?: Record<string, any>;
}

export class VectorPipeline {
  private openai: OpenAI;
  private extractor: PdfExtractor;
  private audioExtractor: AudioExtractor;
  private embeddingModel: string;
  private textSplitter: RecursiveCharacterTextSplitter;

  constructor(options?: {
    embeddingModel?: string;
    chunkSize?: number;
    chunkOverlap?: number;
    extractor?: PdfExtractor;
    audioExtractor?: AudioExtractor;
  }) {
    const apiKey = process.env.API_KEY || process.env.OPENAI_API_KEY;
    const baseURL = process.env.BASE_URL || process.env.OPENAI_BASE_URL;

    if (!apiKey) {
      console.warn('[VectorPipeline] Warning: No API_KEY or OPENAI_API_KEY found in environment.');
    }

    this.openai = new OpenAI({
      apiKey: apiKey || '',
      baseURL: baseURL || undefined
    });

    this.extractor = options?.extractor || pdfExtractor;
    this.audioExtractor = options?.audioExtractor || audioExtractor;
    this.embeddingModel = options?.embeddingModel || process.env.EMBEDDING_MODEL || 'text-embedding-3-small';

    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: options?.chunkSize ?? 800,
      chunkOverlap: options?.chunkOverlap ?? 150
    });
  }

  /**
   * Generates vector embeddings for an array of text strings using OpenAI.
   */
  public async generateEmbeddings(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];

    // Process in batches of 100 to avoid payload size or rate limits
    const batchSize = 100;
    const allEmbeddings: number[][] = [];

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const response = await this.openai.embeddings.create({
        model: this.embeddingModel,
        input: batch
      });

      for (const item of response.data) {
        allEmbeddings.push(item.embedding);
      }
    }

    return allEmbeddings;
  }

  /**
   * Splits extracted PDF pages into clean overlapping text chunks with page tracking.
   */
  public async chunkPdfPages(extraction: PdfExtractionResult): Promise<ChunkWithMetadata[]> {
    const chunks: ChunkWithMetadata[] = [];
    const sourceHash = crypto.createHash('md5').update(extraction.source).digest('hex').slice(0, 8);

    for (const page of extraction.pages) {
      if (!page.text || page.text.trim().length === 0) continue;

      const pageChunks = await this.textSplitter.splitText(page.text);

      pageChunks.forEach((chunkText: string, idx: number) => {
        const chunkId = `doc_${sourceHash}_p${page.pageNumber}_c${idx}_${crypto.randomBytes(4).toString('hex')}`;
        chunks.push({
          id: chunkId,
          text: chunkText,
          pageNumber: page.pageNumber,
          chunkIndex: idx,
          source: extraction.source,
          mediaType: 'pdf',
          metadata: {
            ...extraction.metadata,
            pdfType: extraction.pdfType
          }
        });
      });
    }

    return chunks;
  }

  /**
   * Groups timestamped audio/video segments into coherent passages with exact time labels
   * (e.g. "01:20 - 02:45") for high-quality semantic vector search.
   */
  public async chunkAudioSegments(extraction: AudioExtractionResult): Promise<ChunkWithMetadata[]> {
    const chunks: ChunkWithMetadata[] = [];
    const sourceHash = crypto.createHash('md5').update(extraction.source).digest('hex').slice(0, 8);
    const targetChunkChars = 750;

    const segments = extraction.segments || [];
    let currentPassageLines: string[] = [];
    let currentChars = 0;
    let passageStart = 0;
    let passageEnd = 0;
    let isPassageOpen = false;

    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i]!;
      const segText = seg.text.trim();
      if (!segText) continue;

      if (!isPassageOpen) {
        passageStart = seg.startTimeSeconds;
        passageEnd = seg.endTimeSeconds;
        isPassageOpen = true;
      }

      const speakerPrefix = seg.speakerId !== undefined ? `[Speaker ${seg.speakerId}]: ` : '';
      const line = `[${seg.timeLabel}] ${speakerPrefix}${segText}`;
      currentPassageLines.push(line);
      currentChars += line.length;
      passageEnd = seg.endTimeSeconds;

      // When the accumulated passage reaches the target chunk length or at the end
      if (currentChars >= targetChunkChars || i === segments.length - 1) {
        const fullPassage = currentPassageLines.join('\n');
        const timeLabel = `${formatTimestamp(passageStart)} - ${formatTimestamp(passageEnd)}`;

        // If the passage text is within bounds, make it one chunk
        if (fullPassage.length <= 1200) {
          const chunkId = `aud_${sourceHash}_c${chunks.length}_${crypto.randomBytes(4).toString('hex')}`;
          chunks.push({
            id: chunkId,
            text: fullPassage,
            startTimeSeconds: passageStart,
            endTimeSeconds: passageEnd,
            timeLabel,
            chunkIndex: chunks.length,
            source: extraction.source,
            mediaType: extraction.mediaType,
            metadata: {
              ...extraction.metadata,
              mediaType: extraction.mediaType,
              timeLabel,
              startTimeSeconds: passageStart,
              endTimeSeconds: passageEnd,
              durationSeconds: extraction.durationSeconds,
              languageCode: extraction.detectedLanguage
            }
          });
        } else {
          // If massive passage, split into chunks using textSplitter
          const splitTexts = await this.textSplitter.splitText(fullPassage);
          for (let sIdx = 0; sIdx < splitTexts.length; sIdx++) {
            const chunkId = `aud_${sourceHash}_c${chunks.length}_s${sIdx}_${crypto.randomBytes(4).toString('hex')}`;
            chunks.push({
              id: chunkId,
              text: splitTexts[sIdx]!,
              startTimeSeconds: passageStart,
              endTimeSeconds: passageEnd,
              timeLabel,
              chunkIndex: chunks.length,
              source: extraction.source,
              mediaType: extraction.mediaType,
              metadata: {
                ...extraction.metadata,
                mediaType: extraction.mediaType,
                timeLabel,
                startTimeSeconds: passageStart,
                endTimeSeconds: passageEnd,
                durationSeconds: extraction.durationSeconds,
                languageCode: extraction.detectedLanguage
              }
            });
          }
        }

        currentPassageLines = [];
        currentChars = 0;
        isPassageOpen = false;
      }
    }

    return chunks;
  }

  /**
   * Full PDF ingestion pipeline:
   * 1. Extracts PDF text from URL or local file path
   * 2. Chunks the text using recursive character splitting
   * 3. Generates OpenAI vector embeddings
   * 4. Upserts records into Pinecone vector index
   */
  public async ingestPdf(
    source: string,
    options?: { namespace?: string; indexName?: string }
  ): Promise<IngestionResult> {
    try {
      console.log(`📄 [VectorPipeline] Starting PDF extraction from: ${source}`);
      const extraction = await this.extractor.extract(source);

      if (!extraction.success) {
        return {
          success: false,
          source,
          pdfType: extraction.pdfType,
          totalPages: 0,
          totalChunks: 0,
          vectorsUploaded: 0,
          error: `Extraction failed: ${extraction.error}`
        };
      }

      console.log(`✂️ [VectorPipeline] Chunking ${extraction.totalPages} pages (${extraction.pdfType})...`);
      const chunks = await this.chunkPdfPages(extraction);

      if (chunks.length === 0) {
        return {
          success: false,
          source,
          pdfType: extraction.pdfType,
          totalPages: extraction.totalPages,
          totalChunks: 0,
          vectorsUploaded: 0,
          message: 'No extractable text found in PDF to embed. If this is a scanned PDF, OCR is required.'
        };
      }

      console.log(`🧠 [VectorPipeline] Generating embeddings for ${chunks.length} chunks via ${this.embeddingModel}...`);
      const textsToEmbed = chunks.map((c) => c.text);
      const embeddings = await this.generateEmbeddings(textsToEmbed);

      console.log(`🌲 [VectorPipeline] Upserting vectors into Pinecone...`);
      const index = getPineconeIndex(options?.indexName);
      const targetIndex = options?.namespace ? index.namespace(options.namespace) : index;

      const records = chunks.map((chunk, idx) => {
        const metadata: Record<string, any> = {
          text: chunk.text,
          source: chunk.source,
          chunkIndex: chunk.chunkIndex,
          pdfType: extraction.pdfType,
          mediaType: 'pdf',
          uploadedAt: new Date().toISOString()
        };
        if (chunk.pageNumber !== undefined) {
          metadata.pageNumber = chunk.pageNumber;
        }
        return {
          id: chunk.id,
          values: embeddings[idx]!,
          metadata
        };
      });

      // Upsert in batches of 100
      const upsertBatchSize = 100;
      for (let i = 0; i < records.length; i += upsertBatchSize) {
        const batch = records.slice(i, i + upsertBatchSize);
        await targetIndex.upsert({ records: batch });
      }

      console.log(`✅ [VectorPipeline] Successfully uploaded ${records.length} vectors to Pinecone.`);

      return {
        success: true,
        source,
        pdfType: extraction.pdfType,
        totalPages: extraction.totalPages,
        totalChunks: chunks.length,
        vectorsUploaded: records.length,
        namespace: options?.namespace,
        message: `Successfully processed and indexed ${chunks.length} chunks into Pinecone.`
      };
    } catch (err: any) {
      console.error(`❌ [VectorPipeline] Ingestion error:`, err);
      return {
        success: false,
        source,
        pdfType: 'unknown',
        totalPages: 0,
        totalChunks: 0,
        vectorsUploaded: 0,
        error: err?.message || String(err)
      };
    }
  }

  /**
   * Full Audio/Video transcription & vector ingestion pipeline:
   * 1. Extracts & normalizes audio (converts video via FFmpeg if necessary)
   * 2. Slices massive audio into chunks (~50 mins each)
   * 3. Transcribes using Sarvam AI Batch STT Job API with exact phrase & speaker timestamps
   * 4. Groups segments into coherent passages with time labels (e.g. "01:25 - 03:40")
   * 5. Generates OpenAI vector embeddings
   * 6. Upserts records into Pinecone vector DB
   */
  public async ingestAudio(
    source: string,
    options?: {
      namespace?: string;
      indexName?: string;
      model?: string;
      mode?: 'transcribe' | 'translate' | 'verbatim' | 'translit' | 'codemix';
      withDiarization?: boolean;
      languageCode?: string;
      prompt?: string;
    }
  ): Promise<AudioIngestionResult> {
    try {
      console.log(`🎙️ [VectorPipeline] Starting audio extraction from: ${source}`);
      const extraction = await this.audioExtractor.extract(source, {
        model: options?.model,
        mode: options?.mode,
        withDiarization: options?.withDiarization,
        languageCode: options?.languageCode,
        prompt: options?.prompt
      });

      if (!extraction.success) {
        return {
          success: false,
          source,
          mediaType: extraction.mediaType,
          durationSeconds: 0,
          totalSegments: 0,
          totalChunks: 0,
          vectorsUploaded: 0,
          error: `Audio transcription failed: ${extraction.error}`
        };
      }

      console.log(
        `✂️ [VectorPipeline] Chunking ${extraction.segments.length} timestamped audio segments into vector passages...`
      );
      const chunks = await this.chunkAudioSegments(extraction);

      if (chunks.length === 0) {
        return {
          success: false,
          source,
          mediaType: extraction.mediaType,
          durationSeconds: extraction.durationSeconds,
          totalSegments: extraction.segments.length,
          totalChunks: 0,
          vectorsUploaded: 0,
          message: 'No speech/transcript found in media to embed.'
        };
      }

      console.log(`🧠 [VectorPipeline] Generating embeddings for ${chunks.length} chunks via ${this.embeddingModel}...`);
      const textsToEmbed = chunks.map((c) => c.text);
      const embeddings = await this.generateEmbeddings(textsToEmbed);

      console.log(`🌲 [VectorPipeline] Upserting audio transcript vectors into Pinecone...`);
      const index = getPineconeIndex(options?.indexName);
      const targetIndex = options?.namespace ? index.namespace(options.namespace) : index;

      const records = chunks.map((chunk, idx) => {
        const metadata: Record<string, any> = {
          text: chunk.text,
          source: chunk.source,
          mediaType: extraction.mediaType,
          durationSeconds: extraction.durationSeconds,
          chunkIndex: chunk.chunkIndex,
          uploadedAt: new Date().toISOString()
        };
        if (chunk.timeLabel) {
          metadata.timeLabel = chunk.timeLabel;
        }
        if (chunk.startTimeSeconds !== undefined) {
          metadata.startTimeSeconds = chunk.startTimeSeconds;
        }
        if (chunk.endTimeSeconds !== undefined) {
          metadata.endTimeSeconds = chunk.endTimeSeconds;
        }
        if (extraction.detectedLanguage) {
          metadata.languageCode = extraction.detectedLanguage;
        }
        return {
          id: chunk.id,
          values: embeddings[idx]!,
          metadata
        };
      });

      // Upsert in batches of 100
      const upsertBatchSize = 100;
      for (let i = 0; i < records.length; i += upsertBatchSize) {
        const batch = records.slice(i, i + upsertBatchSize);
        await targetIndex.upsert({ records: batch });
      }

      console.log(`✅ [VectorPipeline] Successfully uploaded ${records.length} audio transcript vectors to Pinecone.`);

      return {
        success: true,
        source,
        mediaType: extraction.mediaType,
        durationSeconds: extraction.durationSeconds,
        totalSegments: extraction.segments.length,
        totalChunks: chunks.length,
        vectorsUploaded: records.length,
        detectedLanguage: extraction.detectedLanguage,
        jobId: extraction.jobId,
        namespace: options?.namespace,
        message: `Successfully transcribed and indexed ${chunks.length} chunks (${formatTimestamp(extraction.durationSeconds)} ${extraction.mediaType}) into Pinecone.`
      };
    } catch (err: any) {
      console.error(`❌ [VectorPipeline] Audio ingestion error:`, err);
      return {
        success: false,
        source,
        mediaType: 'unknown',
        durationSeconds: 0,
        totalSegments: 0,
        totalChunks: 0,
        vectorsUploaded: 0,
        error: err?.message || String(err)
      };
    }
  }

  /**
   * Queries Pinecone for semantically relevant chunks matching a search query.
   */
  public async search(
    query: string,
    options?: { topK?: number; namespace?: string; indexName?: string }
  ): Promise<VectorSearchResult[]> {
    const topK = options?.topK ?? 5;
    const queryEmbeddings = await this.generateEmbeddings([query]);
    const queryVector = queryEmbeddings[0];

    if (!queryVector) {
      throw new Error('Failed to generate embedding for the search query.');
    }

    const index = getPineconeIndex(options?.indexName);
    const targetIndex = options?.namespace ? index.namespace(options.namespace) : index;

    const queryResponse = await targetIndex.query({
      vector: queryVector,
      topK,
      includeMetadata: true
    });

    return (queryResponse.matches || []).map((match) => {
      const meta = (match.metadata as Record<string, any>) || {};
      return {
        id: match.id,
        score: match.score ?? 0,
        text: meta.text || '',
        source: meta.source || '',
        pageNumber: meta.pageNumber,
        startTimeSeconds: meta.startTimeSeconds,
        endTimeSeconds: meta.endTimeSeconds,
        timeLabel: meta.timeLabel,
        mediaType: meta.mediaType,
        metadata: meta
      };
    });
  }

  private parseToolInput(input: string): Record<string, any> {
    try {
      const parsed = JSON.parse(input);
      if (typeof parsed === 'object' && parsed !== null) return parsed;
      return { query: String(parsed) };
    } catch {
      return { query: input.trim() };
    }
  }

  /**
   * Returns ITool objects ready to plug into the Vector Master Agent.
   */
  public getItools(): ITool[] {
    return [
      {
        name: 'ingestPdfToVectorDb',
        description:
          'Downloads/loads a PDF from a URL or local file, extracts text, chunks it, embeds it with OpenAI, and stores it in Pinecone.',
        doc: 'ingestPdfToVectorDb({"source": "https://example.com/document.pdf", "namespace": "optional_namespace"})',
        executor: async (input: string): Promise<string> => {
          try {
            const data = this.parseToolInput(input);
            const source = data.source || data.url || data.path || data.query;
            if (!source) {
              return JSON.stringify({ error: 'Missing required field: "source" (URL or file path)' });
            }
            const result = await this.ingestPdf(source, {
              namespace: data.namespace,
              indexName: data.indexName
            });
            return JSON.stringify(result);
          } catch (err: any) {
            return JSON.stringify({ error: err?.message || String(err) });
          }
        }
      },
      {
        name: 'ingestAudioToVectorDb',
        description:
          'Transcribes an audio or video file (or URL) using Sarvam AI Batch STT (handling massive audio chunks up to 55 mins each with timestamps & diarization), generates OpenAI embeddings, and stores the transcribed segments in Pinecone vector DB.',
        doc: 'ingestAudioToVectorDb({"source": "https://example.com/recording.mp3" | "C:/path/to/video.mp4", "namespace": "optional_namespace", "withDiarization": true, "mode": "transcribe"})',
        executor: async (input: string): Promise<string> => {
          try {
            const data = this.parseToolInput(input);
            const source = data.source || data.url || data.path || data.query;
            if (!source) {
              return JSON.stringify({ error: 'Missing required field: "source" (URL or file path)' });
            }
            const result = await this.ingestAudio(source, {
              namespace: data.namespace,
              indexName: data.indexName,
              model: data.model,
              mode: data.mode,
              withDiarization: data.withDiarization ?? data.with_diarization,
              languageCode: data.languageCode || data.language_code,
              prompt: data.prompt
            });
            return JSON.stringify(result);
          } catch (err: any) {
            return JSON.stringify({ error: err?.message || String(err) });
          }
        }
      },
      {
        name: 'queryVectorMemory',
        description:
          'Performs semantic similarity search over indexed PDF documents and audio/video transcripts in Pinecone to recall facts, answers, or context with timestamps.',
        doc: 'queryVectorMemory({"query": "What are the project requirements?", "topK": 5})',
        executor: async (input: string): Promise<string> => {
          try {
            const data = this.parseToolInput(input);
            const query = data.query || data.search || data.question;
            if (!query) {
              return JSON.stringify({ error: 'Missing required field: "query"' });
            }
            const results = await this.search(query, {
              topK: data.topK ? Number(data.topK) : 5,
              namespace: data.namespace,
              indexName: data.indexName
            });
            return JSON.stringify({ query, count: results.length, matches: results });
          } catch (err: any) {
            return JSON.stringify({ error: err?.message || String(err) });
          }
        }
      }
    ];
  }
}

// Standalone default instance
export const vectorPipeline = new VectorPipeline();
