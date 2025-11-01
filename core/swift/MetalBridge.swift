import Foundation
import Metal

@_cdecl("run_gpu_thread")
public func runGPUThread(_ threadId: Int32) {
    print("[GPU] Executing thread \(threadId) using Metal backend.")

    guard let device: any MTLDevice = MTLCreateSystemDefaultDevice(),
          let commandQueue: any MTLCommandQueue = device.makeCommandQueue() else {
        print("Metal unavailable.")
        return
    }
    // Metal Shader
    let source = """
    kernel void gpuKernel(device float* buffer [[buffer(0)]],
                          uint id [[thread_position_in_grid]]) {
        buffer[id] = buffer[id] + 1.0;
    }
    """

    do {
        let lib: any MTLLibrary = try device.makeLibrary(source: source, options: nil)
        let fn: any MTLFunction = lib.makeFunction(name: "gpuKernel")!
        let pipeline: any MTLComputePipelineState = try device.makeComputePipelineState(function: fn)

        let count: Int = 256
        let buffer: any MTLBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: [])!
        let ptr: UnsafeMutablePointer<Float> = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i: Int in 0..<count { ptr[i] = Float(i) }

        let commandBuffer: any MTLCommandBuffer = commandQueue.makeCommandBuffer()!
        let encoder: any MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)

        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        print("[GPU] Thread \(threadId) complete.")
    } catch {
        print("Metal error: \\(error)")
    }
}