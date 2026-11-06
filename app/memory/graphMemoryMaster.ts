import type { Driver } from 'neo4j-driver';
import { driver as defaultDriver } from './neo4jDriver';
import { Agent, type ITool } from '../agents/agents';
import { GraphMemory } from './memoryFunctions';

// This is the agent that is gonna deal with graph memory.

// Central system prompt for this agent specifically.
const SYSTEM_PROMPT = `
  
`;

// Main class 

export class GraphMaster {
  private agent: Agent;
  private memory: GraphMemory;

  constructor(driver: Driver = defaultDriver) {
    this.memory = new GraphMemory(driver);
    const builder = Agent.builder().setIntructions(SYSTEM_PROMPT);
    for (const tool of this.memory.getItools()) {
      builder.tool(tool);
    }
    this.agent = builder.build();
  }

  // Directly calling the GraphMaster
  public async run(query: string) {
    return await this.agent.run(query);
  }

  // Main agent as tool for Master Agent
  public asTool(): ITool {
    return {
      name: "graphMemoryMaster",
      description: `Fully equipped specialist agent capable of managing and querying the Neo4j Knowledge Graph.
Use this tool when:
1. The user mentions personal facts, relationships, preferences, friends, or entities to persist in long-term memory.
2. You need relational context or past information before answering a question or making a recommendation.
3. You need to extract, query, update, or remove connections between people, places, or things.`,
      doc: 'graphMemoryMaster({"task": "Detailed instruction describing facts to remember or relationships/context to look up"})',
      executor: async (input: string): Promise<string> => {
        let task = input;
        try {
          const parsed = JSON.parse(input);
          task = parsed.task || parsed.query || input;
        } catch {
          // input was a plain string
        }

        // Run the memory specialist's own reasoning loop
        const history = await this.agent.run(task);

        if (!history || history.length === 0) {
          return JSON.stringify({ status: 'error', message: 'No response from memory agent' });
        }

        const lastMessage = history[history.length - 1];
        if (!lastMessage) {
          return JSON.stringify({ status: 'error', message: 'Empty message content from memory agent' });
        }

        // Return the final message content to the Master Agent
        return lastMessage.content;
      }
    };
  }
}


