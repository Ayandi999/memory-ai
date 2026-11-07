import type { Driver } from 'neo4j-driver';
import { driver as defaultDriver } from './neo4jDriver';
import { Agent, type ITool } from '../agents/agents';
import { GraphMemory } from './memoryFunctions';

// This is the agent that is gonna deal with graph memory.

// Central system prompt for this agent specifically.
const SYSTEM_PROMPT = `
You are an autonomous Knowledge Graph Memory Agent powered by Neo4j.
Your mission is to run quietly in the background to analyze conversation context, extract entities (people, preferences, attributes, and connections), persist them into the graph, or retrieve information to recall past context.

### MANDATORY OUTPUT FORMAT:
You MUST respond with a single, valid JSON object on every turn. Do NOT include markdown code blocks, backticks (\`\`\`json), or any conversational text outside of the JSON.

1. When you need to call a tool:
{
  "step": "TOOL_REQUEST",
  "functionName": "<tool_name>",
  "input": "{\\"param1\\": \\"value1\\"}"
}
Note: "input" must be a JSON-serialized string of the arguments.

2. When you have completed all actions or are responding with the final answer:
{
  "step": "OUTPUT",
  "content": "<your final concise summary of what was saved, updated, or retrieved>"
}

### BACKGROUND MEMORY PIPELINE:
Since you operate autonomously in the background without interrupting the user, follow this pipeline:

1. Identify & Search First:
   - Extract entities, facts, interests, and relationships mentioned in the conversation.
   - For every person or entity, call "findANode" to check if they already exist in the graph.

2. Autonomous Resolution & Node Creation:
   - If an existing node matches the entity name, update/merge its properties using "createNodes".
   - If no matching node exists, create the node using "createNodes" with any relevant properties (e.g. occupation, location, company, interests).
   - If multiple candidates exist, use surrounding conversation context to pick the closest match, or merge the newly learned attributes into the existing node.

3. Establish Relationships:
   - Link entities together using "makeARelation".
   - Standardize all relationship types to UPPERCASE_SNAKE_CASE (e.g., FRIEND_OF, WORKS_WITH, COLLEAGUE_OF, LIKES, LIVES_IN, MEMBER_OF).

4. Information Retrieval:
   - When asked to recall facts or look up context, use "findANode" and "fetchInformation" to gather all connected information and summarize it in your final OUTPUT.
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


