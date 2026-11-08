import OpenAI from 'openai';
import { getPineconeIndex } from './pineconeClient';
import 'dotenv/config';

export interface TranscriptSegment {
  text: string;
  start: number; // in seconds (already relative to full video start)
  end: number;   // in seconds
}

export interface TranscriptChunk {
  id: string; // f"{video_id}-{chunk_index}"
  text: string;
  start: number;
  end: number;
  chunkIndex: number;
  videoId: string;
  videoUrl: string;
}

export interface VideoSource {
  video_id: string;
  label: string;
  start: number;
  end: number;
  url: string;
}

export interface IngestResult {
  success: boolean;
  videoId: string;
  totalChunks: number;
  message: string;
}

export interface AskResult {
  answer: string;
  sources: VideoSource[];
}

/**
 * Lazy singleton for OpenAI client using existing environment credentials
 */
let openaiInstance: OpenAI | null = null;
export function getOpenAI(): OpenAI {
  if (openaiInstance) return openaiInstance;

  const apiKey = process.env.API_KEY || process.env.OPENAI_API_KEY;
  const baseURL = process.env.BASE_URL || process.env.OPENAI_BASE_URL;

  openaiInstance = new OpenAI({
    apiKey: apiKey || '',
    baseURL: baseURL || undefined
  });
  return openaiInstance;
}

/**
 * Counts words in a string splitting on whitespace.
 */
export function countWords(str: string): number {
  return str.trim().split(/\s+/).filter(Boolean).length;
}

/**
 * Formats seconds into mm:ss (or h:mm:ss if >= 3600 seconds)
 */
export function formatTimestampLabel(seconds: number): string {
  const totalSecs = Math.max(0, Math.floor(seconds));
  const hrs = Math.floor(totalSecs / 3600);
  const mins = Math.floor((totalSecs % 3600) / 60);
  const secs = totalSecs % 60;
  const pad = (n: number) => (n < 10 ? `0${n}` : `${n}`);

  if (hrs > 0) {
    return `${hrs}:${pad(mins)}:${pad(secs)}`;
  }
  return `${pad(mins)}:${pad(secs)}`;
}

/**
 * Generates video player URL with jump parameter (YouTube ?t=seconds or &t=seconds)
 */
export function formatVideoUrl(baseUrl: string, startSeconds: number): string {
  const intSecs = Math.floor(startSeconds);
  const separator = baseUrl.includes('?') ? '&' : '?';
  return `${baseUrl}${separator}t=${intSecs}`;
}

/**
 * Step 1: Chunking
 * - Groups consecutive segments into chunks of ~150 to 250 words
 * - Overlaps consecutive chunks by exactly 1 segment
 * - start = start of first segment in chunk
 * - end = end of last segment in chunk
 * - id = `${videoId}-${chunkIndex}`
 */
export function chunkTranscriptSegments(
  videoId: string,
  videoUrl: string,
  segments: TranscriptSegment[],
  minWords = 150,
  maxWords = 250
): TranscriptChunk[] {
  if (!segments || segments.length === 0) return [];

  const chunks: TranscriptChunk[] = [];
  let startIndex = 0;

  while (startIndex < segments.length) {
    let currentWords = 0;
    let endIndex = startIndex;

    while (endIndex < segments.length) {
      const segWords = countWords(segments[endIndex]!.text);
      currentWords += segWords;

      // Stop once we accumulate at least minWords, or reach the end of the video
      if (currentWords >= minWords || endIndex === segments.length - 1) {
        break;
      }
      endIndex++;
    }

    const chunkSegments = segments.slice(startIndex, endIndex + 1);
    const chunkText = chunkSegments
      .map((s) => s.text.trim())
      .filter(Boolean)
      .join(' ');
    const chunkStart = chunkSegments[0]!.start;
    const chunkEnd = chunkSegments[chunkSegments.length - 1]!.end;

    chunks.push({
      id: `${videoId}-${chunks.length}`,
      text: chunkText,
      start: chunkStart,
      end: chunkEnd,
      chunkIndex: chunks.length,
      videoId,
      videoUrl
    });

    if (endIndex >= segments.length - 1) {
      break;
    }

    // 1-segment overlap: start the next chunk from the last segment of the current chunk
    startIndex = endIndex > startIndex ? endIndex : startIndex + 1;
  }

  return chunks;
}

/**
 * Generates vector embeddings using the configured embedding model
 */
export async function generateEmbeddings(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];
  const openai = getOpenAI();
  const model = process.env.EMBEDDING_MODEL || 'text-embedding-3-small';

  const batchSize = 100;
  const allEmbeddings: number[][] = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const response = await openai.embeddings.create({
      model,
      input: batch
    });

    for (const item of response.data) {
      allEmbeddings.push(item.embedding);
    }
  }

  return allEmbeddings;
}

/**
 * Deduplicates sources whose time ranges overlap or are adjacent, and merges them into one range.
 */
export function mergeAndDeduplicateSources(
  sources: Array<{ video_id: string; video_url: string; start: number; end: number }>
): VideoSource[] {
  if (sources.length === 0) return [];

  // Group by video_id
  const byVideo = new Map<string, Array<{ video_id: string; video_url: string; start: number; end: number }>>();
  for (const s of sources) {
    const list = byVideo.get(s.video_id) || [];
    list.push(s);
    byVideo.set(s.video_id, list);
  }

  const mergedResults: VideoSource[] = [];

  for (const [videoId, ranges] of byVideo.entries()) {
    // Sort chronologically by start
    ranges.sort((a, b) => a.start - b.start);

    const merged: Array<{ start: number; end: number; video_url: string }> = [];

    for (const r of ranges) {
      if (merged.length === 0) {
        merged.push({ start: r.start, end: r.end, video_url: r.video_url });
        continue;
      }

      const prev = merged[merged.length - 1]!;
      // Overlap or adjacency check: if next start is within previous end + 2 seconds tolerance
      if (r.start <= prev.end + 2) {
        prev.end = Math.max(prev.end, r.end);
      } else {
        merged.push({ start: r.start, end: r.end, video_url: r.video_url });
      }
    }

    for (const m of merged) {
      mergedResults.push({
        video_id: videoId,
        label: `${formatTimestampLabel(m.start)} to ${formatTimestampLabel(m.end)}`,
        start: Number(m.start.toFixed(1)),
        end: Number(m.end.toFixed(1)),
        url: formatVideoUrl(m.video_url, m.start)
      });
    }
  }

  return mergedResults;
}

/**
 * DELIVERABLE 1: ingest(video_id, video_url, segments)
 * Stores transcript chunks into Pinecone idempotently.
 */
export async function ingest(
  videoId: string,
  videoUrl: string,
  segments: TranscriptSegment[],
  options?: { namespace?: string; indexName?: string }
): Promise<IngestResult> {
  const chunks = chunkTranscriptSegments(videoId, videoUrl, segments);
  if (chunks.length === 0) {
    return {
      success: false,
      videoId,
      totalChunks: 0,
      message: 'No valid transcript segments provided.'
    };
  }

  // 1. Generate embeddings
  const texts = chunks.map((c) => c.text);
  const embeddings = await generateEmbeddings(texts);

  // 2. Connect to Pinecone Index
  const index = getPineconeIndex(options?.indexName);
  const targetIndex = options?.namespace ? index.namespace(options.namespace) : index;

  // 3. Make ingestion idempotent: delete old chunks for this video_id first
  try {
    await targetIndex.deleteMany({ filter: { video_id: { $eq: videoId } } });
  } catch (err) {
    // Pinecone serverless may require id list; upsert deterministic IDs f"{videoId}-{chunk_index}" overwrites existing
    console.warn(`[ingest] deleteMany note: ${(err as any)?.message || err}`);
  }

  // 4. Upsert records into vector DB
  const records = chunks.map((chunk, idx) => ({
    id: chunk.id, // f"{video_id}-{chunk_index}"
    values: embeddings[idx]!,
    metadata: {
      video_id: chunk.videoId,
      video_url: chunk.videoUrl,
      chunk_index: chunk.chunkIndex,
      start: chunk.start, // numeric seconds
      end: chunk.end,     // numeric seconds
      text: chunk.text
    }
  }));

  const batchSize = 100;
  for (let i = 0; i < records.length; i += batchSize) {
    const batch = records.slice(i, i + batchSize);
    await targetIndex.upsert({ records: batch });
  }

  return {
    success: true,
    videoId,
    totalChunks: chunks.length,
    message: `Successfully ingested ${chunks.length} chunks for video "${videoId}".`
  };
}

/**
 * DELIVERABLE 2: ask(question, video_id=None)
 * Retrieves top chunks, prompts LLM for answer with citations, and maps back to exact timestamps.
 */
export async function ask(
  question: string,
  videoId?: string,
  options?: { namespace?: string; indexName?: string; topK?: number }
): Promise<AskResult> {
  const topK = options?.topK ?? 5;

  // 1. Embed user's question
  const [queryVector] = await generateEmbeddings([question]);
  if (!queryVector) {
    throw new Error('Failed to generate embedding for the search question.');
  }

  // 2. Retrieve top chunks from Pinecone
  const index = getPineconeIndex(options?.indexName);
  const targetIndex = options?.namespace ? index.namespace(options.namespace) : index;

  const queryParams: any = {
    vector: queryVector,
    topK,
    includeMetadata: true
  };
  if (videoId) {
    queryParams.filter = { video_id: { $eq: videoId } };
  }

  const queryResponse = await targetIndex.query(queryParams);
  const matches = queryResponse.matches || [];

  if (matches.length === 0) {
    return {
      answer: "I couldn't find any relevant transcript sections to answer this question.",
      sources: []
    };
  }

  // 3. Build LLM context in similarity order as numbered excerpts: "[1] text...", "[2] text..."
  const excerptsMap = new Map<
    number,
    { videoId: string; videoUrl: string; start: number; end: number; text: string }
  >();

  const contextLines: string[] = [];
  matches.forEach((match, idx) => {
    const excerptNum = idx + 1;
    const meta = (match.metadata || {}) as Record<string, any>;
    excerptsMap.set(excerptNum, {
      videoId: String(meta.video_id || ''),
      videoUrl: String(meta.video_url || ''),
      start: Number(meta.start || 0),
      end: Number(meta.end || 0),
      text: String(meta.text || '')
    });
    contextLines.push(`[${excerptNum}] ${meta.text || ''}`);
  });

  // 4. Prompt the LLM strictly with instructions to cite excerpt numbers only (no hallucinated timestamps)
  const prompt = `You are an assistant answering user questions based exclusively on video transcripts.

EXCERPTS FROM TRANSCRIPT:
${contextLines.join('\n\n')}

QUESTION:
${question}

INSTRUCTIONS:
1. Answer using ONLY these excerpts.
2. Cite the excerpt numbers you used like [1], [2].
3. DO NOT invent or mention timestamps yourself.
4. If the answer is not in the excerpts, state clearly: "I couldn't find the answer in the transcript."

ANSWER:`;

  const openai = getOpenAI();
  const completion = await openai.chat.completions.create({
    model: process.env.CHAT_MODEL || 'gpt-4o-mini',
    messages: [{ role: 'user', content: prompt }],
    temperature: 0
  });

  const answer = completion.choices[0]?.message?.content?.trim() || '';

  // 5. Parse cited excerpt numbers from LLM answer (e.g. "[1]", "[2]")
  const matchesIter = answer.matchAll(/\[(\d+)\]/g);
  const citedExcerptNumbers = new Set<number>();
  for (const m of matchesIter) {
    const num = parseInt(m[1]!, 10);
    if (excerptsMap.has(num)) {
      citedExcerptNumbers.add(num);
    }
  }

  // Fallback: If LLM produced an answer without brackets, cite excerpt [1]
  if (
    citedExcerptNumbers.size === 0 &&
    !answer.toLowerCase().includes("couldn't find the answer") &&
    matches.length > 0
  ) {
    citedExcerptNumbers.add(1);
  }

  // 6. Look up start/end from metadata of cited chunks (LLM never produces timestamps)
  const rawSources: Array<{ video_id: string; video_url: string; start: number; end: number }> = [];
  for (const num of citedExcerptNumbers) {
    const item = excerptsMap.get(num);
    if (item && item.videoId) {
      rawSources.push({
        video_id: item.videoId,
        video_url: item.videoUrl,
        start: item.start,
        end: item.end
      });
    }
  }

  // 7. Deduplicate & merge overlapping/adjacent ranges
  const sources = mergeAndDeduplicateSources(rawSources);

  return {
    answer,
    sources
  };
}
