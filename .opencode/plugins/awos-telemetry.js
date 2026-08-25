export const AWOSTelemetry = async ({ project, client, $, directory, worktree }) => {
  console.log("[AWOS Telemetry] Tracking session activity")

  return {
    event: async ({ event }) => {
      if (event.type === "session.idle") {
        console.log("[AWOS Telemetry] Session completed — ready for checkpoint")
      }
      if (event.type === "session.error") {
        console.error("[AWOS Telemetry] Session error:", event.error?.message || "unknown")
      }
    },

    "tool.execute.after": async (input, output) => {
      const tool = input.tool
      if (["edit", "write", "bash"].includes(tool)) {
        console.log(`[AWOS Telemetry] Tool executed: ${tool}`)
      }
    },
  }
}
