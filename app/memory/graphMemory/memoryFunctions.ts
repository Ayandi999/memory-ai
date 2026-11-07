import type { Driver } from 'neo4j-driver';
import type { ITool } from '../../agents/agents.ts';

export interface PersonProperties {
  name: string;
  [key: string]: any;
}

export class GraphMemory {
  private driver: Driver;

  constructor(driver: Driver) {
    this.driver = driver;
  }

  // Initialise fulltext index on Person for fuzzy search
  public async initIndex() {
    await this.executeCypher(`
      CREATE FULLTEXT INDEX personNameIndex IF NOT EXISTS
      FOR (p:Person) ON EACH [p.name]
    `);
    console.log('✅ Graph index initialised.');
  }

  private async executeCypher<T = any>(
    query: string,
    params: Record<string, any> = {}
  ): Promise<T[]> {
    const session = this.driver.session();
    try {
      const result = await session.run(query, params);
      return result.records.map((r) => r.toObject() as T);
    } finally {
      await session.close();
    }
  }

  // Create or update a Person node
  public async createNodes(name: string, properties: Record<string, any> = {}) {
    const query = `
      MERGE (p:Person {name: $name})
      SET p += $properties
      RETURN p
    `;
    const normalizedName = name.toLowerCase().trim();
    const records = await this.executeCypher(query, { name: normalizedName, properties });
    return records[0];
  }

  // Find details about a person based on fuzzy similarities using the fulltext index
  public async findANode(name: string) {
    const cypher = `
      CALL db.index.fulltext.queryNodes("personNameIndex", $searchTerm)
      YIELD node, score
      RETURN node.name AS name, properties(node) AS properties, score
      ORDER BY score DESC
      LIMIT 5
    `;
    // Lucene fuzzy operator: appending '~' enables Levenshtein distance matching
    const searchTerm = `${name.toLowerCase().trim()}~`;
    return await this.executeCypher(cypher, { searchTerm });
  }

  // Create a directed relation between two persons
  public async makeARelation(
    fromName: string,
    toName: string,
    relationshipType: string,
    properties: Record<string, any> = {}
  ) {
    const sanitizedRelation = relationshipType.toUpperCase().replace(/[^A-Z_]/g, '_');
    const query = `
      MATCH (a:Person {name: $fromName})
      MATCH (b:Person {name: $toName})
      MERGE (a)-[r:${sanitizedRelation}]->(b)
      SET r += $properties
      RETURN a.name AS from, type(r) AS relation, b.name AS to, properties(r) AS relationProperties
    `;
    return await this.executeCypher(query, {
      fromName: fromName.toLowerCase().trim(),
      toName: toName.toLowerCase().trim(),
      properties
    });
  }

  // Fetch all known connections and facts related to a person
  public async fetchInformation(name: string) {
    const query = `
      MATCH (p:Person {name: $name})-[r]-(target)
      RETURN p.name AS person, type(r) AS relationship, properties(r) AS relationProperties, target.name AS connectedTo, properties(target) AS targetProperties
    `;
    return await this.executeCypher(query, { name: name.toLowerCase().trim() });
  }

  //Remove a node ad all relations assossiated with it detach delete
  public async deleteNode(name: string) {
    const query = `
      MATCH (p:Person {name:$name})
      DETACH DELETE p 
      RETURN count(p) AS deletedCount
    `
    return await this.executeCypher(query, { name: name.toLowerCase().trim() });
  }
  //remove a relation specifically
  public async deleteRelation(fromName: string, toName: string, relationshipType?: string) {
    const query = `
    MATCH (a:Person {name: $fromName})-[r]->(b:Person {name: $toName})
    WHERE $relationshipType IS NULL OR type(r) = $relationshipType
    DELETE r
    RETURN count(r) AS deletedCount
  `;
    return await this.executeCypher(query, {
      fromName: fromName.toLowerCase().trim(),
      toName: toName.toLowerCase().trim(),
      relationshipType: relationshipType ? relationshipType.toUpperCase() : null
    });
  }

  // Helper to safely parse tool inputs that LLMs might pass as JSON strings or raw strings
  private parseToolInput(input: string): Record<string, any> {
    try {
      const parsed = JSON.parse(input);
      if (typeof parsed === 'object' && parsed !== null) return parsed;
      return { query: String(parsed) };
    } catch {
      return { query: input.trim() };
    }
  }

  public getItools(): ITool[] {
    return [
      {
        name: 'createNodes',
        description: 'Creates or updates a Person node with optional attributes/properties in the knowledge graph.',
        doc: 'createNodes({"name": "string", "properties": {"age": 25, "city": "London"}})',
        executor: async (input: string) => {
          try {
            const data = this.parseToolInput(input);
            const name = data.name || data.query;
            if (!name) return JSON.stringify({ error: 'Missing required field: "name"' });
            const result = await this.createNodes(name, data.properties || {});
            return JSON.stringify({ success: true, node: result });
          } catch (err: any) {
            return JSON.stringify({ error: err?.message || String(err) });
          }
        }
      },
      {
        name: 'findANode',
        description: 'Performs a fuzzy search in the knowledge graph to find person nodes matching or similar to a given name.',
        doc: 'findANode({"name": "string"})',
        executor: async (input: string) => {
          try {
            const data = this.parseToolInput(input);
            const name = data.name || data.query;
            if (!name) return JSON.stringify({ error: 'Missing required field: "name"' });
            const result = await this.findANode(name);
            return JSON.stringify({ matches: result });
          } catch (err: any) {
            return JSON.stringify({ error: err?.message || String(err) });
          }
        }
      },
      {
        name: 'makeARelation',
        description: 'Creates a relationship between two persons in the knowledge graph (e.g. FRIEND_OF, LIKES, KNOWS, WORKS_WITH).',
        doc: 'makeARelation({"fromName": "string", "toName": "string", "relationshipType": "string", "properties": {}})',
        executor: async (input: string) => {
          try {
            const data = this.parseToolInput(input);
            const { fromName, toName, relationshipType, properties } = data;
            if (!fromName || !toName || !relationshipType) {
              return JSON.stringify({
                error: 'Missing required fields: "fromName", "toName", and "relationshipType"'
              });
            }
            const result = await this.makeARelation(fromName, toName, relationshipType, properties || {});
            return JSON.stringify({ success: true, relationship: result });
          } catch (err: any) {
            return JSON.stringify({ error: err?.message || String(err) });
          }
        }
      },
      {
        name: 'fetchInformation',
        description: 'Fetches all relationships and connected entities for a specific person to answer questions or recall memory.',
        doc: 'fetchInformation({"name": "string"})',
        executor: async (input: string) => {
          try {
            const data = this.parseToolInput(input);
            const name = data.name || data.query;
            if (!name) return JSON.stringify({ error: 'Missing required field: "name"' });
            const result = await this.fetchInformation(name);
            return JSON.stringify({ person: name, connections: result });
          } catch (err: any) {
            return JSON.stringify({ error: err?.message || String(err) });
          }
        }
      },
      {
        name: 'deleteRelation',
        description: 'Deletes a relationship between two persons in the knowledge graph (optionally filtered by relationshipType).',
        doc: 'deleteRelation({"fromName": "string", "toName": "string", "relationshipType": "string"})',
        executor: async (input: string) => {
          try {
            const data = this.parseToolInput(input);
            const { fromName, toName, relationshipType } = data;
            if (!fromName || !toName) {
              return JSON.stringify({
                error: 'Missing required fields: "fromName" and "toName"'
              });
            }
            const result = await this.deleteRelation(fromName, toName, relationshipType);
            return JSON.stringify({ success: true, result });
          } catch (err: any) {
            return JSON.stringify({ error: err?.message || String(err) });
          }
        }
      },
      {
        name: 'deleteNode',
        description: 'Permanently deletes a person node and detaches all connected relationships from the knowledge graph.',
        doc: 'deleteNode({"name": "string"})',
        executor: async (input: string) => {
          try {
            const data = this.parseToolInput(input);
            const name = data.name || data.query;
            if (!name) return JSON.stringify({ error: 'Missing required field: "name"' });
            const result = await this.deleteNode(name);
            return JSON.stringify({ success: true, result });
          } catch (err: any) {
            return JSON.stringify({ error: err?.message || String(err) });
          }
        }
      }
    ];
  }
}



