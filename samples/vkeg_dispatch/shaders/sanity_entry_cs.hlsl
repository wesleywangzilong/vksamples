struct Particle
{
    float2 position;
    float2 velocity;
    float4 color;
};

struct ParameterUBO
{
    float deltaTime;
};

struct InputPayload
{
    uint3 grid_size : SV_DispatchGrid;
};

ParameterUBO ubo : register(u0);
// [[vk::image_format("rgba8")]] [[vk::binding(0, 0)]] RWTexture2D<float4> outImage;
[[vk::binding(1, 0)]] RWStructuredBuffer<Particle> particlesIn;
[[vk::binding(2, 0)]] RWStructuredBuffer<Particle> particlesOut;

[Shader("node")]
[NodeIsProgramEntry]
[NodeDispatchGrid(64, 64, 1)]
[NumThreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID,
// DispatchNodeInputRecord<Particle> particlesIn
    DispatchNodeInputRecord<InputPayload> in_payload
// ,[MaxRecords(16)] NodeOutput<Particle> particlesOut
)
{
    uint index = DTid.x;

    Particle particleIn = particlesIn[index];

    const float speed = 0.8; // test different speed;
    particlesOut[index].position = particleIn.position + particleIn.velocity.xy * (ubo.deltaTime * speed);
    particlesOut[index].velocity = particleIn.velocity;

    // Flip movement at window border
    if ((particlesOut[index].position.x <= -1.0) || (particlesOut[index].position.x >= 1.0))
    {
        particlesOut[index].velocity.x = -particlesOut[index].velocity.x;
    }
    if ((particlesOut[index].position.y <= -1.0) || (particlesOut[index].position.y >= 1.0))
    {
        particlesOut[index].velocity.y = -particlesOut[index].velocity.y;
    }
}
