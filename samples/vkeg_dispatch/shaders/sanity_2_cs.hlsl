

#define TILE_SIZE 256 * 32

struct Particle
{
    float2 position;
    float2 velocity;
    float4 color;
};

struct Particle_Node
{
    float2 position;
    float2 velocity;
    float4 color;
    uint index;
};

[[vk::binding(1, 0)]] RWStructuredBuffer<Particle> particlesBuffer;
[[vk::binding(2, 0)]] RWStructuredBuffer<Particle> particlesOut;

[Shader("node")]
[NodeID("second")]
[NodeLaunch("broadcasting")]
[NodeDispatchGrid(1, 1, 1)]
[NumThreads(1, 1, 1)]
void main(
    const uint3 DTid : SV_DispatchThreadID,

    const uint svGroupIndex : SV_GroupIndex,
    const uint3 svGroupThreadId : SV_GroupThreadID,
    DispatchNodeInputRecord<Particle_Node> in_payload)
{
    uint index = DTid.x;

    Particle_Node particlesIn = in_payload.Get();

    // test payload transfer
    particlesOut[particlesIn.index].position = particlesIn.position;
    particlesOut[particlesIn.index].velocity = particlesIn.velocity;

    // test binding buffer transfer
    // particlesOut[particlesIn.index].position = particlesBuffer[particlesIn.index].position;
    // particlesOut[particlesIn.index].velocity = particlesBuffer[particlesIn.index].velocity;

}
