import neo4j from 'neo4j-driver';
import { getNeededEnvVariables } from '../../agents/agents';

export const driver = neo4j.driver(
  getNeededEnvVariables('NEO4J_URI'),
  neo4j.auth.basic(
    getNeededEnvVariables('NEO4J_USERNAME'),
    getNeededEnvVariables('NEO4J_PASSWORD')
  )
);

export async function connectToGraphDB() {
  try {
    const serverInfo = await driver.verifyConnectivity();
    console.log('Server is connected successfully');
    return { success: true, serverInfo };
  } catch (error) {
    console.error(`Failed to connect to the server.\n error:${error}`);
    throw error;
  }
}

export async function closeNeo4j() {
  await driver.close();
}


