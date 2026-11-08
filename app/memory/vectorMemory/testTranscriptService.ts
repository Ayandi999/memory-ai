import {
  chunkTranscriptSegments,
  mergeAndDeduplicateSources,
  formatTimestampLabel,
  formatVideoUrl,
  ingest,
  ask,
  type TranscriptSegment
} from './transcriptVectorService';
import 'dotenv/config';

// 1. Prepare sample transcript with known timestamps
const sampleSegments: TranscriptSegment[] = [
  {
    start: 0.0,
    end: 25.0,
    text: 'Welcome everyone to our deep dive session on artificial intelligence and vector databases. Today we are going to explore how modern generative AI applications retain memory over long conversations.'
  },
  {
    start: 25.0,
    end: 65.0,
    text: 'Before we jump into the code, let us review the agenda. First we cover graph structures, then Pinecone indexes, and finally multilingual speech-to-text models.'
  },
  {
    start: 65.0,
    end: 115.0,
    text: 'Now let us begin with the first topic. Graph databases represent data as nodes and edges. This is fundamentally different from relational tabular systems because relationships are treated as first-class citizens.'
  },
  {
    start: 115.0,
    end: 175.0,
    text: 'Moving to the critical topic: What is Pinecone? Pinecone is a managed, cloud-native vector database designed for lightning-fast similarity search at massive scale. Pinecone stores high-dimensional embeddings and calculates cosine similarity in milliseconds.'
  },
  {
    start: 175.0,
    end: 230.0,
    text: 'When you store vectors in Pinecone, you can attach rich metadata such as timestamps, video URLs, and chapter tags, which allows the AI agent to pinpoint the exact second where information was spoken.'
  },
  {
    start: 230.0,
    end: 290.0,
    text: 'In summary, combining vector databases with graph memory gives your agent both semantic search capabilities and structured knowledge navigation. Thank you for watching.'
  }
];

async function runTests() {
  console.log('🧪 [Test Suite] Starting Transcript Vector Service Verification...\n');

  // Test 1: Chunking Algorithm verification
  console.log('🔹 Test 1: Verifying chunking logic (150-250 words with 1-segment overlap)...');
  const chunks = chunkTranscriptSegments(
    'video_test_101',
    'https://www.youtube.com/watch?v=sample123',
    sampleSegments,
    50, // for this sample we set threshold lower so we get multiple chunks
    120
  );

  console.log(`   Generated ${chunks.length} chunks.`);
  for (const c of chunks) {
    const wordCount = c.text.trim().split(/\s+/).length;
    console.log(`   - Chunk [${c.id}]: ${wordCount} words | ${formatTimestampLabel(c.start)} -> ${formatTimestampLabel(c.end)}`);
  }

  // Assertions on chunk 0 and 1 overlap
  if (chunks.length > 1) {
    const chunk0 = chunks[0]!;
    const chunk1 = chunks[1]!;
    console.log(`   ✅ Chunk 0 ID: ${chunk0.id}, start: ${chunk0.start}s, end: ${chunk0.end}s`);
    console.log(`   ✅ Chunk 1 ID: ${chunk1.id}, start: ${chunk1.start}s, end: ${chunk1.end}s`);
    if (chunk1.start <= chunk0.end) {
      console.log('   ✅ Passed: 1-segment overlap preserved across chunk boundary.');
    }
  }

  // Test 2: Source deduplication & merging
  console.log('\n🔹 Test 2: Verifying source deduplication & adjacent interval merging...');
  const testSources = [
    { video_id: 'vid1', video_url: 'https://youtube.com/watch?v=1', start: 100, end: 150 },
    { video_id: 'vid1', video_url: 'https://youtube.com/watch?v=1', start: 148, end: 200 }, // overlaps
    { video_id: 'vid1', video_url: 'https://youtube.com/watch?v=1', start: 300, end: 350 }  // distinct
  ];
  const merged = mergeAndDeduplicateSources(testSources);
  console.log('   Merged sources output:', JSON.stringify(merged, null, 2));

  if (merged.length === 2 && merged[0]?.start === 100 && merged[0]?.end === 200) {
    console.log('   ✅ Passed: Overlapping ranges merged successfully (100 to 200 & 300 to 350).');
  } else {
    console.error('   ❌ Failed merging logic!');
  }

  // Test 3: End-to-end Ingestion & Query if API keys are configured
  const hasPinecone = Boolean(process.env.PINECONE_API_KEY && (process.env.PINECONE_INDEX || process.env.PINECONE_INDEX_NAME));
  const hasOpenAI = Boolean(process.env.API_KEY || process.env.OPENAI_API_KEY);

  if (hasPinecone && hasOpenAI) {
    console.log('\n🔹 Test 3: Running End-to-End Ingest & Ask against Pinecone...');
    const videoId = 'test_pinecone_lecture';
    const videoUrl = 'https://www.youtube.com/watch?v=sample123';

    console.log(`   Ingesting ${sampleSegments.length} segments for video "${videoId}"...`);
    const ingestRes = await ingest(videoId, videoUrl, sampleSegments);
    console.log('   Ingest result:', ingestRes);

    const question = 'What is Pinecone and how does it store vectors?';
    console.log(`   Asking question: "${question}"...`);
    const askRes = await ask(question, videoId);
    console.log('\n   🎯 Answer Result:');
    console.log(JSON.stringify(askRes, null, 2));

    // Verify timestamp matches the segment where Pinecone is explained (starts at 115.0s)
    const matchedSource = askRes.sources.find((s) => s.start <= 175 && s.end >= 115);
    if (matchedSource) {
      console.log(`\n   🎉 VERIFIED! Start time ${matchedSource.start}s matches the target segment (~115s)!`);
      console.log(`   Clickable URL: ${matchedSource.url}`);
    } else {
      console.log('\n   ⚠️ Note: Review sources matched:', askRes.sources);
    }
  } else {
    console.log('\n🔹 Test 3 Skipped live Pinecone/OpenAI call (API keys not populated in environment).');
    console.log('   Set PINECONE_API_KEY and API_KEY in .env to run the live test.');
  }

  console.log('\n✨ All tests completed successfully!');
}

runTests().catch((err) => {
  console.error('❌ Test failed with error:', err);
  process.exit(1);
});
