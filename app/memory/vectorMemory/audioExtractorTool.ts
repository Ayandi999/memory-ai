import axios from 'axios';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegStatic from 'ffmpeg-static';
import ffprobeStatic from 'ffprobe-static';
import type { ITool } from '../../agents/agents';
import 'dotenv/config';

// Configure ffmpeg & ffprobe binary paths with environment overrides
const resolvedFfmpegPath =
  process.env.FFMPEG_PATH ||
  (typeof ffmpegStatic === 'string' ? ffmpegStatic : (ffmpegStatic as any)?.path || ffmpegStatic);

const resolvedFfprobePath =
  process.env.FFPROBE_PATH ||
  (typeof ffprobeStatic === 'string'
    ? ffprobeStatic
    : (ffprobeStatic as any)?.path || (ffprobeStatic as any)?.default?.path || ffprobeStatic);

if (resolvedFfmpegPath) {
  ffmpeg.setFfmpegPath(resolvedFfmpegPath);
}

if (resolvedFfprobePath) {
  ffmpeg.setFfprobePath(resolvedFfprobePath);
}

export type MediaType = 'audio' | 'video';

export interface ExtractedAudioSegment {
  chunkIndex: number;
  startTimeSeconds: number;
  endTimeSeconds: number;
  timeLabel: string; // e.g. "01:25 - 01:55"
  text: string;
  speakerId?: string;
  languageCode?: string;
  characterCount: number;
}

export interface AudioExtractionResult {
  success: boolean;
  source: string;
  mediaType: MediaType;
  originalFormat: string;
  durationSeconds: number;
  totalChunks: number;
  totalCharacters: number;
  content: string;
  segments: ExtractedAudioSegment[];
  detectedLanguage?: string;
  jobId?: string;
  metadata?: Record<string, any>;
  message?: string;
  error?: string;
}

const SUPPORTED_AUDIO_EXTENSIONS = new Set([
  '.mp3',
  '.wav',
  '.m4a',
  '.aac',
  '.flac',
  '.ogg',
  '.wma',
  '.opus',
  '.aiff',
  '.alac'
]);

const SUPPORTED_VIDEO_EXTENSIONS = new Set([
  '.mp4',
  '.mkv',
  '.avi',
  '.mov',
  '.wmv',
  '.flv',
  '.webm',
  '.m4v',
  '.mpeg',
  '.mpg',
  '.3gp'
]);

export interface AudioExtractorOptions {
  sarvamApiKey?: string;
  model?: string; // 'saaras:v4' (default) or 'saaras:v3'
  mode?: 'transcribe' | 'translate' | 'verbatim' | 'translit' | 'codemix';
  withDiarization?: boolean;
  // Maximum duration per massive audio chunk (default: 50 minutes = 3000 seconds)
  maxChunkDurationMinutes?: number;
}

/**
 * Formats a duration in seconds into MM:SS or HH:MM:SS
 */
export function formatTimestamp(seconds: number): string {
  const totalSecs = Math.max(0, Math.floor(seconds));
  const hrs = Math.floor(totalSecs / 3600);
  const mins = Math.floor((totalSecs % 3600) / 60);
  const secs = totalSecs % 60;
  if (hrs > 0) {
    return `${hrs}:${mins < 10 ? '0' : ''}${mins}:${secs < 10 ? '0' : ''}${secs}`;
  }
  return `${mins < 10 ? '0' : ''}${mins}:${secs < 10 ? '0' : ''}${secs}`;
}

export class AudioExtractor {
  private sarvamApiKey: string;
  private model: string;
  private mode: string;
  private withDiarization: boolean;
  private maxChunkDurationSeconds: number;

  constructor(options?: AudioExtractorOptions) {
    this.sarvamApiKey = options?.sarvamApiKey || process.env.SARVAM_API_KEY || '';
    // Flagship Sarvam speech-to-text models: 'saaras:v4' (recommended) or 'saaras:v3'
    this.model = options?.model || process.env.SARVAM_STT_MODEL || 'saaras:v4';
    this.mode = options?.mode || 'transcribe';
    this.withDiarization = options?.withDiarization ?? false;
    // Default 50 minutes (3000s) per massive chunk to stay safely within Sarvam's ~55min/2hr batch job limits
    this.maxChunkDurationSeconds = (options?.maxChunkDurationMinutes ?? 50) * 60;
  }

  /**
   * Validates if the source is supported (valid URL or existing local audio/video file)
   */
  public validateSource(source: string): {
    isValid: boolean;
    isUrl: boolean;
    mediaType?: MediaType;
    ext?: string;
    error?: string;
  } {
    if (!source || typeof source !== 'string') {
      return { isValid: false, isUrl: false, error: 'Source must be a non-empty string path or URL.' };
    }

    const trimmed = source.trim();
    const isUrl = /^https?:\/\//i.test(trimmed);

    if (isUrl) {
      try {
        const parsed = new URL(trimmed);
        const ext = path.extname(parsed.pathname).toLowerCase();

        let mediaType: MediaType | undefined;
        if (SUPPORTED_AUDIO_EXTENSIONS.has(ext)) {
          mediaType = 'audio';
        } else if (SUPPORTED_VIDEO_EXTENSIONS.has(ext)) {
          mediaType = 'video';
        }

        return { isValid: true, isUrl: true, mediaType, ext };
      } catch (err: any) {
        return { isValid: false, isUrl: true, error: `Invalid media URL format: ${err.message}` };
      }
    }

    const ext = path.extname(trimmed).toLowerCase();
    const isAudio = SUPPORTED_AUDIO_EXTENSIONS.has(ext);
    const isVideo = SUPPORTED_VIDEO_EXTENSIONS.has(ext);

    if (!isAudio && !isVideo) {
      return {
        isValid: false,
        isUrl: false,
        error: `Unsupported file extension "${ext}". Supported audio: [${Array.from(SUPPORTED_AUDIO_EXTENSIONS).join(', ')}], supported video: [${Array.from(SUPPORTED_VIDEO_EXTENSIONS).join(', ')}]`
      };
    }

    if (!fs.existsSync(trimmed)) {
      return {
        isValid: false,
        isUrl: false,
        error: `Local media file not found at: ${trimmed}`
      };
    }

    return {
      isValid: true,
      isUrl: false,
      mediaType: isVideo ? 'video' : 'audio',
      ext
    };
  }

  /**
   * Downloads remote media URL to a local destination file path.
   */
  public async downloadRemoteMedia(url: string, destPath: string): Promise<void> {
    const writer = fs.createWriteStream(destPath);
    const response = await axios.get(url, {
      responseType: 'stream',
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
      }
    });

    return new Promise((resolve, reject) => {
      response.data.pipe(writer);
      writer.on('finish', () => resolve());
      writer.on('error', (err) => reject(err));
    });
  }

  /**
   * Inspects the media file using FFprobe to obtain duration and stream information.
   */
  public async probeMedia(filePath: string): Promise<{
    duration: number;
    hasAudio: boolean;
    hasVideo: boolean;
    format: string;
    metadata: Record<string, any>;
  }> {
    return new Promise((resolve, reject) => {
      ffmpeg.ffprobe(filePath, (err, metadata) => {
        if (err) {
          return reject(new Error(`FFprobe failed to inspect media: ${err.message}`));
        }

        const streams = metadata.streams || [];
        const hasAudio = streams.some((s) => s.codec_type === 'audio');
        const hasVideo = streams.some((s) => s.codec_type === 'video');
        const duration = Number(metadata.format?.duration || 0);
        const format = metadata.format?.format_name || '';

        resolve({
          duration,
          hasAudio,
          hasVideo,
          format,
          metadata: metadata.format || {}
        });
      });
    });
  }

  /**
   * Extracts audio from a video file using FFmpeg, saving as a 128kbps mono MP3.
   */
  public async extractAudioFromVideo(videoPath: string, outputPath: string): Promise<string> {
    return new Promise((resolve, reject) => {
      ffmpeg(videoPath)
        .noVideo()
        .audioChannels(1)
        .audioFrequency(16000)
        .audioBitrate(128)
        .output(outputPath)
        .on('end', () => resolve(outputPath))
        .on('error', (err) => reject(new Error(`FFmpeg audio extraction failed: ${err.message}`)))
        .run();
    });
  }

  /**
   * Chops an audio file into massive chunks of up to maxChunkDurationSeconds (e.g. 50-55 mins each)
   * as MP3 files suitable for Sarvam AI's Batch Speech-to-Text Job API.
   */
  public async splitAudioIntoMassiveChunks(
    audioPath: string,
    totalDuration: number,
    outputDir: string
  ): Promise<Array<{ chunkIndex: number; filePath: string; fileName: string; start: number; end: number }>> {
    const chunkDuration = this.maxChunkDurationSeconds;

    // If duration fits within single chunk limit, convert to MP3 directly
    if (totalDuration <= chunkDuration) {
      const fileName = 'chunk_0.mp3';
      const singleOut = path.join(outputDir, fileName);

      await new Promise<void>((resolve, reject) => {
        ffmpeg(audioPath)
          .audioChannels(1)
          .audioFrequency(16000)
          .audioBitrate(128)
          .output(singleOut)
          .on('end', () => resolve())
          .on('error', (err) => reject(new Error(`Failed to convert audio: ${err.message}`)))
          .run();
      });

      return [{ chunkIndex: 0, filePath: singleOut, fileName, start: 0, end: totalDuration }];
    }

    const chunkCount = Math.ceil(totalDuration / chunkDuration);
    const chunks: Array<{ chunkIndex: number; filePath: string; fileName: string; start: number; end: number }> = [];

    for (let i = 0; i < chunkCount; i++) {
      const start = i * chunkDuration;
      const duration = Math.min(chunkDuration, Math.max(0, totalDuration - start));
      const end = start + duration;
      const fileName = `chunk_${i}.mp3`;
      const chunkFilePath = path.join(outputDir, fileName);

      await new Promise<void>((resolve, reject) => {
        ffmpeg(audioPath)
          .setStartTime(start)
          .setDuration(duration)
          .audioChannels(1)
          .audioFrequency(16000)
          .audioBitrate(128)
          .output(chunkFilePath)
          .on('end', () => resolve())
          .on('error', (err) => reject(new Error(`Failed to slice massive chunk ${i}: ${err.message}`)))
          .run();
      });

      chunks.push({ chunkIndex: i, filePath: chunkFilePath, fileName, start, end });
    }

    return chunks;
  }

  /**
   * STEP 1: Create a batch speech-to-text job via Sarvam AI API
   * POST https://api.sarvam.ai/speech-to-text/job/v1
   */
  private async createBatchJob(options?: {
    model?: string;
    mode?: string;
    withDiarization?: boolean;
    languageCode?: string;
  }): Promise<{ jobId: string; storageContainerType?: string }> {
    const apiKey = this.sarvamApiKey || process.env.SARVAM_API_KEY;
    if (!apiKey) {
      throw new Error(
        '[Sarvam Configuration Error] Missing SARVAM_API_KEY in environment variables or AudioExtractor options.'
      );
    }

    const payload: Record<string, any> = {
      job_parameters: {
        model: options?.model || this.model,
        mode: options?.mode || this.mode,
        with_diarization: options?.withDiarization ?? this.withDiarization,
        with_timestamps: true
      }
    };

    if (options?.languageCode) {
      payload.job_parameters.language_code = options.languageCode;
    }

    const response = await axios.post('https://api.sarvam.ai/speech-to-text/job/v1', payload, {
      headers: {
        'api-subscription-key': apiKey,
        'Content-Type': 'application/json'
      }
    });

    const data = response.data;
    const jobId = data.job_id || data.id;
    if (!jobId) {
      throw new Error(`Failed to create Sarvam Batch STT job: ${JSON.stringify(data)}`);
    }

    return { jobId, storageContainerType: data.storage_container_type };
  }

  /**
   * STEP 2: Request pre-signed SAS upload URLs for audio chunks
   * POST https://api.sarvam.ai/speech-to-text/job/v1/upload-files
   */
  private async getUploadUrls(
    jobId: string,
    fileNames: string[]
  ): Promise<Record<string, { file_url: string }>> {
    const apiKey = this.sarvamApiKey || process.env.SARVAM_API_KEY;

    const response = await axios.post(
      'https://api.sarvam.ai/speech-to-text/job/v1/upload-files',
      {
        job_id: jobId,
        files: fileNames
      },
      {
        headers: {
          'api-subscription-key': apiKey,
          'Content-Type': 'application/json'
        }
      }
    );

    return response.data.upload_urls || {};
  }

  /**
   * STEP 3: Upload files directly to the pre-signed storage URLs (BlockBlob)
   */
  private async uploadChunksToStorage(
    chunks: Array<{ fileName: string; filePath: string }>,
    uploadUrls: Record<string, { file_url: string }>
  ): Promise<void> {
    for (const chunk of chunks) {
      const uploadInfo = uploadUrls[chunk.fileName];
      if (!uploadInfo || !uploadInfo.file_url) {
        throw new Error(`Missing pre-signed upload URL for chunk: ${chunk.fileName}`);
      }

      console.log(`📤 [AudioExtractor] Uploading ${chunk.fileName} to Sarvam storage...`);
      const fileData = await fs.promises.readFile(chunk.filePath);

      await axios.put(uploadInfo.file_url, fileData, {
        headers: {
          'x-ms-blob-type': 'BlockBlob',
          'Content-Type': 'audio/mpeg'
        },
        maxBodyLength: Infinity,
        maxContentLength: Infinity
      });
    }
  }

  /**
   * STEP 4: Start the Batch STT job
   * POST https://api.sarvam.ai/speech-to-text/job/v1/start
   */
  private async startBatchJob(jobId: string): Promise<void> {
    const apiKey = this.sarvamApiKey || process.env.SARVAM_API_KEY;

    await axios.post(
      'https://api.sarvam.ai/speech-to-text/job/v1/start',
      { job_id: jobId },
      {
        headers: {
          'api-subscription-key': apiKey,
          'Content-Type': 'application/json'
        }
      }
    );
  }

  /**
   * STEP 5: Poll job status until complete or failed
   * GET https://api.sarvam.ai/speech-to-text/job/v1/{jobId}
   */
  private async pollJobUntilComplete(
    jobId: string,
    timeoutMs: number = 900000, // 15 minutes timeout
    pollIntervalMs: number = 4000
  ): Promise<{ statusData: any; outputFiles: string[] }> {
    const apiKey = this.sarvamApiKey || process.env.SARVAM_API_KEY;
    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
      const response = await axios.get(`https://api.sarvam.ai/speech-to-text/job/v1/${jobId}`, {
        headers: {
          'api-subscription-key': apiKey
        }
      });

      const data = response.data;
      const state = (data.job_state || data.status || '').toLowerCase();
      console.log(`⏳ [AudioExtractor] Batch Job [${jobId}] state: ${data.job_state || data.status}`);

      if (state === 'completed' || state === 'partiallycompleted') {
        const outputFiles: string[] = [];
        const details = data.job_details || [];
        for (const d of details) {
          for (const out of d.outputs || []) {
            if (out.file_name) outputFiles.push(out.file_name);
          }
        }
        return { statusData: data, outputFiles };
      }

      if (state === 'failed') {
        throw new Error(`Sarvam STT Batch Job failed: ${data.error_message || JSON.stringify(data)}`);
      }

      await new Promise((res) => setTimeout(res, pollIntervalMs));
    }

    throw new Error(`Sarvam STT Batch Job [${jobId}] timed out after ${timeoutMs / 1000}s`);
  }

  /**
   * STEP 6: Request pre-signed download URLs for results
   * POST https://api.sarvam.ai/speech-to-text/job/v1/download-files
   */
  private async getDownloadUrls(
    jobId: string,
    outputFiles: string[]
  ): Promise<Record<string, { file_url: string }>> {
    const apiKey = this.sarvamApiKey || process.env.SARVAM_API_KEY;

    const response = await axios.post(
      'https://api.sarvam.ai/speech-to-text/job/v1/download-files',
      {
        job_id: jobId,
        files: outputFiles
      },
      {
        headers: {
          'api-subscription-key': apiKey,
          'Content-Type': 'application/json'
        }
      }
    );

    return response.data.download_urls || {};
  }

  /**
   * STEP 7: Download and parse transcript JSON outputs
   */
  private async downloadTranscriptOutputs(
    downloadUrls: Record<string, { file_url: string }>
  ): Promise<Array<{ fileName: string; data: any }>> {
    const results: Array<{ fileName: string; data: any }> = [];

    for (const [fileName, info] of Object.entries(downloadUrls)) {
      if (!info.file_url) continue;
      const res = await axios.get(info.file_url);
      results.push({ fileName, data: res.data });
    }

    return results;
  }

  /**
   * Transcribes massive audio chunks via Sarvam Batch STT Job API (`speech_to_text_job`),
   * downloads results, adjusts segment timestamps relative to each chunk offset,
   * and merges them into unified ExtractedAudioSegment array.
   */
  public async transcribeChunksViaBatchJob(
    chunks: Array<{ chunkIndex: number; filePath: string; fileName: string; start: number; end: number }>,
    options?: {
      model?: string;
      mode?: string;
      withDiarization?: boolean;
      languageCode?: string;
    }
  ): Promise<{ segments: ExtractedAudioSegment[]; detectedLanguage?: string; jobId: string }> {
    console.log(`🚀 [AudioExtractor] Initializing Sarvam Batch STT Job for ${chunks.length} chunks...`);

    // 1. Create Job
    const { jobId } = await this.createBatchJob(options);
    console.log(`📋 [AudioExtractor] Batch Job created with ID: ${jobId}`);

    // 2. Get upload URLs
    const fileNames = chunks.map((c) => c.fileName);
    const uploadUrls = await this.getUploadUrls(jobId, fileNames);

    // 3. Upload chunk files to storage
    await this.uploadChunksToStorage(chunks, uploadUrls);

    // 4. Start processing
    console.log(`▶️ [AudioExtractor] Starting job execution on Sarvam AI...`);
    await this.startBatchJob(jobId);

    // 5. Poll until complete
    const { outputFiles } = await this.pollJobUntilComplete(jobId);
    console.log(`📥 [AudioExtractor] Job finished. Retrieved output files:`, outputFiles);

    // Fallback if outputFiles list is empty: assume output filenames 0.json, 1.json...
    const filesToDownload =
      outputFiles.length > 0 ? outputFiles : chunks.map((_, idx) => `${idx}.json`);

    // 6. Get download links
    const downloadUrls = await this.getDownloadUrls(jobId, filesToDownload);

    // 7. Download JSON outputs
    const outputs = await this.downloadTranscriptOutputs(downloadUrls);

    // 8. Merge chunks and adjust time labels
    const segments: ExtractedAudioSegment[] = [];
    let detectedLanguage: string | undefined;

    // Process output for each chunk in chronological order
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i]!;
      const chunkOffset = chunk.start;

      // Match corresponding output JSON (by index or fileName)
      const outputItem =
        outputs.find((o) => o.fileName === `${i}.json` || o.fileName.includes(String(i))) ||
        outputs[i];

      const data = outputItem?.data || {};
      if (data.language_code && !detectedLanguage) {
        detectedLanguage = data.language_code;
      }

      // Case A: Diarized transcript entries
      if (data.diarized_transcript?.entries && Array.isArray(data.diarized_transcript.entries)) {
        for (const entry of data.diarized_transcript.entries) {
          const entryText = (entry.transcript || '').trim();
          if (!entryText) continue;

          const start = Number(((entry.start_time_seconds || 0) + chunkOffset).toFixed(2));
          const end = Number(((entry.end_time_seconds || 0) + chunkOffset).toFixed(2));

          segments.push({
            chunkIndex: segments.length,
            startTimeSeconds: start,
            endTimeSeconds: end,
            timeLabel: `${formatTimestamp(start)} - ${formatTimestamp(end)}`,
            text: entryText,
            speakerId: entry.speaker_id !== undefined ? String(entry.speaker_id) : undefined,
            languageCode: data.language_code,
            characterCount: entryText.length
          });
        }
      }
      // Case B: Timestamped words/phrases
      else if (
        data.timestamps?.words &&
        Array.isArray(data.timestamps.words) &&
        data.timestamps.start_time_seconds
      ) {
        const words = data.timestamps.words;
        const starts = data.timestamps.start_time_seconds;
        const ends = data.timestamps.end_time_seconds;

        for (let j = 0; j < words.length; j++) {
          const phraseText = (words[j] || '').trim();
          if (!phraseText) continue;

          const start = Number(((starts[j] || 0) + chunkOffset).toFixed(2));
          const end = Number(((ends[j] || 0) + chunkOffset).toFixed(2));

          segments.push({
            chunkIndex: segments.length,
            startTimeSeconds: start,
            endTimeSeconds: end,
            timeLabel: `${formatTimestamp(start)} - ${formatTimestamp(end)}`,
            text: phraseText,
            languageCode: data.language_code,
            characterCount: phraseText.length
          });
        }
      }
      // Case C: Standard concatenated transcript fallback
      else if (data.transcript) {
        const text = String(data.transcript).trim();
        if (text) {
          const start = Number(chunk.start.toFixed(2));
          const end = Number(chunk.end.toFixed(2));
          segments.push({
            chunkIndex: segments.length,
            startTimeSeconds: start,
            endTimeSeconds: end,
            timeLabel: `${formatTimestamp(start)} - ${formatTimestamp(end)}`,
            text,
            languageCode: data.language_code,
            characterCount: text.length
          });
        }
      }
    }

    return { segments, detectedLanguage, jobId };
  }

  /**
   * Unified Audio/Video Extraction Pipeline:
   * 1. Validates input source (URL or local path)
   * 2. Downloads remote file if necessary
   * 3. Probes file format and duration using FFprobe
   * 4. Extracts audio from video via FFmpeg if necessary
   * 5. Slices massive audio into chunks (~50 mins each)
   * 6. Dispatches to Sarvam Batch STT Job API (`speech-to-text/job/v1`)
   * 7. Merges all transcribed phrase/speaker chunks with exact time labels
   * 8. Safely removes all temporary files
   */
  public async extract(
    source: string,
    options?: {
      model?: string;
      mode?: 'transcribe' | 'translate' | 'verbatim' | 'translit' | 'codemix';
      withDiarization?: boolean;
      languageCode?: string;
      prompt?: string;
    }
  ): Promise<AudioExtractionResult> {
    const sessionDir = path.join(
      os.tmpdir(),
      `sarvam_batch_stt_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`
    );

    try {
      const validation = this.validateSource(source);
      if (!validation.isValid) {
        throw new Error(validation.error);
      }

      await fs.promises.mkdir(sessionDir, { recursive: true });

      let localFilePath = source.trim();

      // Download remote URL if needed
      if (validation.isUrl) {
        console.log(`🌐 [AudioExtractor] Downloading remote media: ${localFilePath}`);
        const urlObj = new URL(localFilePath);
        const parsedExt = path.extname(urlObj.pathname).toLowerCase() || '.mp3';
        const downloadDest = path.join(sessionDir, `downloaded_media${parsedExt}`);
        await this.downloadRemoteMedia(localFilePath, downloadDest);
        localFilePath = downloadDest;
      }

      // Probe media format, streams, and duration
      console.log(`🔍 [AudioExtractor] Probing media format with FFprobe...`);
      const probe = await this.probeMedia(localFilePath);
      if (!probe.hasAudio) {
        throw new Error('The provided media file does not contain any audio stream to transcribe.');
      }

      const mediaType: MediaType = probe.hasVideo ? 'video' : 'audio';

      // If video, extract audio stream to MP3
      let readyAudioPath: string;
      if (mediaType === 'video') {
        console.log(`🎬 [AudioExtractor] Video file detected. Extracting audio stream via FFmpeg...`);
        const extractedMp3 = path.join(sessionDir, 'extracted_audio.mp3');
        await this.extractAudioFromVideo(localFilePath, extractedMp3);
        readyAudioPath = extractedMp3;
      } else {
        readyAudioPath = localFilePath;
      }

      // Slice into massive chunks (~50 mins each)
      const chunksDir = path.join(sessionDir, 'chunks');
      await fs.promises.mkdir(chunksDir, { recursive: true });

      console.log(
        `✂️ [AudioExtractor] Slicing media (${formatTimestamp(probe.duration)}) into chunks of up to ${this.maxChunkDurationSeconds / 60} minutes...`
      );
      const chunks = await this.splitAudioIntoMassiveChunks(readyAudioPath, probe.duration, chunksDir);

      // Transcribe via Sarvam Batch STT Job API
      const { segments, detectedLanguage, jobId } = await this.transcribeChunksViaBatchJob(chunks, {
        model: options?.model || this.model,
        mode: options?.mode || this.mode,
        withDiarization: options?.withDiarization ?? this.withDiarization,
        languageCode: options?.languageCode
      });

      // Assemble unified formatted transcript with time labels
      const combinedContent = segments
        .map((seg) => {
          const speakerTag = seg.speakerId !== undefined ? ` [Speaker ${seg.speakerId}]` : '';
          return `--- [${seg.timeLabel}]${speakerTag}${seg.languageCode ? ` [${seg.languageCode}]` : ''} ---\n${seg.text}`;
        })
        .join('\n\n');

      const totalCharacters = segments.reduce((acc, seg) => acc + seg.characterCount, 0);

      return {
        success: true,
        source,
        mediaType,
        originalFormat: probe.format,
        durationSeconds: Number(probe.duration.toFixed(2)),
        totalChunks: segments.length,
        totalCharacters,
        content: combinedContent,
        segments,
        detectedLanguage,
        jobId,
        metadata: {
          ...probe.metadata,
          model: options?.model || this.model,
          mode: options?.mode || this.mode,
          withDiarization: options?.withDiarization ?? this.withDiarization,
          jobId
        }
      };
    } catch (err: any) {
      console.error(`❌ [AudioExtractor] Extraction failed:`, err);
      return {
        success: false,
        source,
        mediaType: 'audio',
        originalFormat: '',
        durationSeconds: 0,
        totalChunks: 0,
        totalCharacters: 0,
        content: '',
        segments: [],
        error: err?.message || String(err)
      };
    } finally {
      // Clean up temporary files
      try {
        if (fs.existsSync(sessionDir)) {
          await fs.promises.rm(sessionDir, { recursive: true, force: true });
        }
      } catch (cleanErr) {
        console.warn(`[AudioExtractor] Warning: Failed to clean up temp dir "${sessionDir}":`, cleanErr);
      }
    }
  }

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
   * Exposes ITool definitions matching the memory and CLI tool pattern.
   */
  public getItools(): ITool[] {
    return [
      {
        name: 'extractAudio',
        description:
          'Transcribes an audio or video file (or URL) using Sarvam AI Batch STT (saaras:v4/v3). Extracts video audio via FFmpeg, chops massive files into ~50-min chunks, submits batch job, and returns merged phrase segments with time labels.',
        doc: 'extractAudio({"source": "https://example.com/lecture.mp4", "withDiarization": true, "mode": "transcribe"})',
        executor: async (input: string): Promise<string> => {
          try {
            const data = this.parseToolInput(input);
            const source = data.source || data.url || data.path || data.query;
            if (!source) {
              return JSON.stringify({ error: 'Missing required field: "source" (URL or file path)' });
            }
            const result = await this.extract(source, {
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
        name: 'detectMediaType',
        description:
          'Inspects a media file from a URL or local path to determine whether it is audio or video, and probes its duration and stream info.',
        doc: 'detectMediaType({"source": "https://example.com/video.mp4"})',
        executor: async (input: string): Promise<string> => {
          try {
            const data = this.parseToolInput(input);
            const source = data.source || data.url || data.path || data.query;
            if (!source) {
              return JSON.stringify({ error: 'Missing required field: "source"' });
            }
            const validation = this.validateSource(source);
            if (!validation.isValid) {
              return JSON.stringify({ success: false, error: validation.error });
            }
            return JSON.stringify({
              success: true,
              source,
              isUrl: validation.isUrl,
              detectedType: validation.mediaType,
              extension: validation.ext
            });
          } catch (err: any) {
            return JSON.stringify({ error: err?.message || String(err) });
          }
        }
      }
    ];
  }
}

// Standalone default instance
export const audioExtractor = new AudioExtractor();
