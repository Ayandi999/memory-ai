import { Pinecone } from '@pinecone-database/pinecone';
import 'dotenv/config';

let pineconeInstance: Pinecone | null = null;

export function getPineconeClient(): Pinecone {
  if (pineconeInstance) return pineconeInstance;

  const apiKey = process.env.PINECONE_API_KEY?.trim();
  if (!apiKey) {
    throw new Error('[Pinecone Configuration Error] Missing PINECONE_API_KEY in environment variables.');
  }

  pineconeInstance = new Pinecone({ apiKey });
  return pineconeInstance;
}

export function getPineconeIndex(indexName?: string) {
  const targetIndex = indexName || process.env.PINECONE_INDEX?.trim() || process.env.PINECONE_INDEX_NAME?.trim();
  if (!targetIndex) {
    throw new Error('[Pinecone Configuration Error] Missing PINECONE_INDEX (or PINECONE_INDEX_NAME) in environment variables.');
  }

  const client = getPineconeClient();
  return client.index(targetIndex);
}

export async function verifyPineconeConnection(indexName?: string): Promise<{ success: boolean; indexes: string[]; targetIndex: string }> {
  try {
    const client = getPineconeClient();
    const indexList = await client.listIndexes();
    const targetIndex = indexName || process.env.PINECONE_INDEX?.trim() || '';
    const indexNames = indexList.indexes?.map((i) => i.name) || [];
    
    console.log(`✅ Connected to Pinecone. Available indexes: ${indexNames.join(', ') || 'None'}`);
    return {
      success: true,
      indexes: indexNames,
      targetIndex
    };
  } catch (error: any) {
    console.error(`❌ Failed to connect to Pinecone: ${error?.message || error}`);
    throw error;
  }
}
