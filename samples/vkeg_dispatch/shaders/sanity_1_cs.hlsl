#define TILE_SIZE	    16
#define MAX_ALLOCATIONS 256	// TILE_SIZE*TILE_SIZE

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
[[vk::binding(2, 0)]] RWStructuredBuffer<Particle> particlesOutBuffer;

[Shader("node")]
[NodeIsProgramEntry]
[NodeDispatchGrid(32, 1, 1)]
[NumThreads(256, 1, 1)]
void main(

    uint3 DTid : SV_DispatchThreadID,
// DispatchNodeInputRecord<Particle> particlesIn
    DispatchNodeInputRecord<InputPayload> in_payload, // how to buffer this???
// ,[MaxRecords(16)] NodeOutput<Particle> particlesOut

    [MaxRecords(MAX_ALLOCATIONS)]
    [NodeID("second")]
    NodeOutput<Particle_Node> particlesOut

    // [MaxRecords(MAX_ALLOCATIONS)]
    // [NodeID("second_2")]
    // NodeOutput<Particle_Node> particlesOut2
)
{
    uint index = DTid.x;
    // We only have one array entry per node and always allocate 1 record per invocation or workgroup.
    const uint recordCount = 1; // what is this???

    const float speed = 0.8; // test different speed;

    Particle particleIn = particlesIn[index];

    ThreadNodeOutputRecords<Particle_Node> out_payload_a = particlesOut.GetThreadNodeOutputRecords(recordCount);

    out_payload_a[0].position = particleIn.position + particleIn.velocity.xy * (ubo.deltaTime * speed);
    out_payload_a[0].velocity = particleIn.velocity;
    out_payload_a[0].index = index;

    // Flip movement at window border
    if ((out_payload_a[0].position.x <= -1.0) || (out_payload_a[0].position.x >= 1.0))
    {
        out_payload_a[0].velocity.x = -out_payload_a[0].velocity.x;
    }
    if ((out_payload_a[0].position.y <= -1.0) || (out_payload_a[0].position.y >= 1.0))
    {
        out_payload_a[0].velocity.y = -out_payload_a[0].velocity.y;
    }

    // test buffer transfer
    // particleIn is local var
    particlesIn[index].position = out_payload_a[0].position;
    particlesIn[index].velocity = out_payload_a[0].velocity;

    // particlesOutBuffer[index].position = out_payload_a[0].position;
    // particlesOutBuffer[index].velocity = out_payload_a[0].velocity;

    out_payload_a.OutputComplete();
}
