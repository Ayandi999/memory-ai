import { exec } from 'child_process';

export interface ITool {
  name: string;
  description: string;
  doc?: string;
  executor: (input: string) => Promise<string>;
}

export class CliTools {
  /**
   * Executes a CLI command on the host system.
   */
  public async executeCommand(cmd: string): Promise<string> {
    return new Promise((resolve) => {
      exec(cmd, (err, stdout, stderr) => {
        if (err) {
          return resolve(`There was an Error: ${err.message}\nStderr: ${stderr}`);
        }
        return resolve(stdout || 'Command executed successfully with no output.');
      });
    });
  }

  /**
   * Returns ITool definitions matching the memory pattern
   */
  public getItools(): ITool[] {
    return [
      {
        name: 'execCli',
        description: 'Runs a CLI command on the users machine and returns stdout output.',
        doc: 'execCli({"command": "string"}) or execCli("string command")',
        executor: async (input: string) => {
          let command = input;
          try {
            const parsed = JSON.parse(input);
            if (typeof parsed === 'object' && parsed !== null && parsed.command) {
              command = parsed.command;
            } else if (typeof parsed === 'string') {
              command = parsed;
            }
          } catch {
            // raw string command
          }
          return await this.executeCommand(command);
        }
      }
    ];
  }
}

// Standalone instance & default exported tool for convenience
export const cliToolsManager = new CliTools();
export const cliAccessTool: ITool | undefined = cliToolsManager.getItools()[0];
