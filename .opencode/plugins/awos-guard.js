export const AWOSGuard = async ({ project, client, $, directory, worktree }) => {
  console.log("[AWOS Guard] Plugin loaded — protecting .awosignore and enforcing AWOS conventions")

  return {
    "tool.execute.before": async (input, output) => {
      if (input.tool === "edit" || input.tool === "write") {
        const path = output.args?.filePath || output.args?.path || ""
        if (path.includes(".awosignore")) {
          throw new Error(
            "AWOS Guard: .awosignore files are protected. " +
            "To modify them, explicitly confirm this is intentional."
          )
        }
      }

      if (input.tool === "bash" && output.args?.command) {
        const cmd = output.args.command
        if (cmd.includes("rm -rf /") || cmd.includes("sudo rm")) {
          throw new Error("AWOS Guard: destructive system-wide operations blocked")
        }
      }
    },

    "shell.env": async (input, output) => {
      output.env.AWOS_ROOT = directory
      output.env.AWOS_PYTHON = "python3.10"
    },
  }
}
