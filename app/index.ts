
import { Agent } from './agents/agents.js';
import { cliAccessTool } from './agents/tools.js';
import { connectToGraphDB } from './memory/graphMemory/neo4jDriver.js';

async function init() {
  // Initialising memory:
  const { serverInfo } = await connectToGraphDB();

  // Here Agent.builder() returns an agent builder class object,
  // then we do method chaining on the builder,
  // and on calling .build() it returns the Agent instance.
  const agent: Agent = Agent.builder()
    .setIntructions(`You are an expert coding agent`)
    .tool(cliAccessTool!)
    .build();

  agent.attachInterceptor(message => console.log(`Message: ${message.role}: ${message.content}`));

  const result = await agent.run('can you build a simple hello world program in c++ on my current project as hello.cpp');
  console.log(result![result?.length! - 1]);
}

init();

