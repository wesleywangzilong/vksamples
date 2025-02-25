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


ParameterUBO ubo : register(u0);;
[[vk::binding(1, 0)]] RWStructuredBuffer<Particle> particlesIn;
[[vk::binding(2, 0)]] RWStructuredBuffer<Particle> particlesOut;

[numthreads(256, 1, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
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