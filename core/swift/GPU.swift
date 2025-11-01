import Foundation
import Metal

enum GPUCommand: String {
    case pause, kill, reboot, sync
}

func log(_ message: String) {
    let timestamp = ISO8601DateFormatter().string(from: Date())
    print("[\(timestamp)] \(message)")
}

struct GPUThreadManager {
    static func handle(command: GPUCommand, threadID: String) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            log("ERROR: Failed to initialize Metal device or command queue.")
            return
        }

        // Create a small control buffer shared with GPU
        let bufferLength = MemoryLayout<UInt32>.size
        guard let controlBuffer = device.makeBuffer(length: bufferLength, options: .storageModeShared) else {
            log("ERROR: Unable to allocate control buffer.")
            return
        }

        let pointer = controlBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)

        switch command {
        case .pause:
            pointer.pointee = 1 // Convention: 1 = PAUSE
            log("PAUSE: GPU Thread \(threadID) flagged with PAUSE=1.")
        case .kill:
            pointer.pointee = 2 // 2 = KILL
            log("KILL: GPU Thread \(threadID) flagged with KILL=2.")
        case .reboot:
            pointer.pointee = 3 // 3 = REBOOT
            log("REBOOT: GPU Thread \(threadID) flagged with REBOOT=3.")
        case .sync:
            pointer.pointee = 4 // 4 = SYNC
            log("SYNC: GPU Thread \(threadID) flagged with SYNC=4.")
        }

        // Optionally enqueue a dummy command buffer to synchronize with GPU
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            commandBuffer.label = "ControlSignal-\(command)"
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
    }
}

// Entry point
let args = CommandLine.arguments
guard args.count == 3,
      let command = GPUCommand(rawValue: args[1].lowercased()) else {
    print("Usage: GPUThreadManager.swift <command> <threadID>")
    print("Commands: pause, kill, reboot, sync")
    exit(1)
}

let threadID = args[2]
GPUThreadManager.handle(command: command, threadID: threadID)
