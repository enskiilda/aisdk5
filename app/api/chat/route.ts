import { anthropic } from "@ai-sdk/anthropic";
import { streamText, convertToModelMessages, stepCountIs } from "ai";
import type { UIMessage } from "ai";
import { killDesktop } from "@/lib/e2b/utils";
import { bashTool, computerTool } from "@/lib/e2b/tool";

// Allow streaming responses up to 30 seconds
export const maxDuration = 300;

export async function POST(req: Request) {
  const { messages, sandboxId }: { messages: UIMessage[]; sandboxId: string } =
    await req.json();
  try {
    // Convert UIMessages to ModelMessages for streamText
    const modelMessages = convertToModelMessages(messages);
    
    const result = streamText({
      model: anthropic("claude-3-7-sonnet-20250219"), // Using Sonnet for computer use
      system:
        "You are a helpful assistant with access to a computer. " +
        "Use the computer tool to help the user with their requests. " +
        "Use the bash tool to execute commands on the computer. You can create files and folders using the bash tool. Always prefer the bash tool where it is viable for the task. " +
        "Be sure to advise the user when waiting is necessary. " +
        "If the browser opens with a setup wizard, YOU MUST IGNORE IT and move straight to the next step (e.g. input the url in the search bar).",
      messages: modelMessages,
      tools: { computer: computerTool(sandboxId), bash: bashTool(sandboxId) },
      // In SDK 5, use stopWhen with stepCountIs instead of maxSteps
      stopWhen: stepCountIs(30),
      providerOptions: {
        anthropic: { cacheControl: { type: "ephemeral" } },
      },
    });

    // In AI SDK 5, use toUIMessageStreamResponse for chat UI messages
    return result.toUIMessageStreamResponse();
  } catch (error) {
    console.error("Chat API error:", error);
    await killDesktop(sandboxId); // Force cleanup on error
    return new Response(JSON.stringify({ error: "Internal Server Error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
