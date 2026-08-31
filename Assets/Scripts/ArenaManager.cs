using Unity.MLAgents;
using UnityEngine;
public class ArenaManager : MonoBehaviour
{
    [Header("Agents")]
    public ShooterAgent agentA;
    public ShooterAgent agentB;

    [Header("Spawns")]
    public Transform spawnA;
    public Transform spawnB;
    public bool randomiseSpawns = true;
    public float spawnPositionJitter = 2f;
    public float spawnYawJitter = 30f;

    //opposite spawns in a circle
    public bool radialSpawns = true;

    //spawndistances
    public float spawnRadius = 20f;

    [Header("Centre block (curriculum)")]
    public Transform centreBlock;
    public float blockHeight = 3f;
    public float defaultBlockSize = 12f;
    const string BlockSizeParameter = "block_size";

    [Header("Cheats (curriculum)")]
    [Range(0f, 1f)]
    public float defaultCheatRate = 0.03f;
    public bool allowMultipleCheats = false;
    public Vector2 aimSnapRange = new Vector2(0.2f, 1f);
    public Vector2 triggerBotRange = new Vector2(0.2f, 1f);
    const float WallhackSeverity = 1f;
    const string CheatRateParameter = "cheat_rate";
    const string MultiCheatParameter = "multi_cheat";

    [Header("Arena bounds")]
    public Vector2 arenaSize = new Vector2(50f, 50f);

    [Header("Episode")]
    public int maxEpisodeSteps = 3000;

    [Header("Terminal rewards")]
    public float winReward = 1f;
    public float loseReward = -1f;
    public float drawReward = -1f;

    bool roundOver;

    public float MaxDistance => arenaSize.magnitude;

    public int MaxEpisodeSteps => maxEpisodeSteps;

    void Start()
    {
        ResetArena();
    }

    void FixedUpdate()
    {
        if (roundOver) return;

        if (agentA.StepCount >= maxEpisodeSteps || agentB.StepCount >= maxEpisodeSteps)
            EndRoundDraw();
    }

    public void ReportDeath(ShooterAgent loser)
    {
        if (roundOver) return;
        roundOver = true;

        ShooterAgent winner = GetOpponent(loser);

        winner.AddReward(winReward);
        loser.AddReward(loseReward);

        agentA.EndEpisode();
        agentB.EndEpisode();

        ResetArena();
    }

    void EndRoundDraw()
    {
        roundOver = true;

        agentA.AddReward(drawReward);
        agentB.AddReward(drawReward);

        //interupt instead of terminate
        agentA.EpisodeInterrupted();
        agentB.EpisodeInterrupted();

        ResetArena();
    }

    void ApplyBlockSize()
    {
        if (centreBlock == null) return;

        float size = Academy.Instance.EnvironmentParameters.GetWithDefault(BlockSizeParameter, defaultBlockSize);
        bool hasBlock = size > 0.01f;
        centreBlock.gameObject.SetActive(hasBlock);

        if (hasBlock)
            centreBlock.localScale = new Vector3(size, blockHeight, size);
    }
    void AssignCheats(ShooterAgent agent, float cheatRate, bool multiCheat)
    {
        float wallhack = 0f;
        float aimSnap = 0f;
        float triggerBot = 0f;

        if (Random.value < cheatRate)
        {
            if (multiCheat)
            {
                if (Random.value < 0.5f) wallhack = WallhackSeverity;
                if (Random.value < 0.5f) aimSnap = RandomIn(aimSnapRange);
                if (Random.value < 0.5f) triggerBot = RandomIn(triggerBotRange);

                if (wallhack == 0f && aimSnap == 0f && triggerBot == 0f)
                    aimSnap = RandomIn(aimSnapRange);
            }
            else
            {
                switch (Random.Range(0, 3))
                {
                    case 0: wallhack = WallhackSeverity; break;
                    case 1: aimSnap = RandomIn(aimSnapRange); break;
                    default: triggerBot = RandomIn(triggerBotRange); break;
                }
            }
        }

        agent.ApplyCheatConfig(wallhack, aimSnap, triggerBot);
    }

    static float RandomIn(Vector2 range)
    {
        float min = Mathf.Clamp(Mathf.Min(range.x, range.y), 0.01f, 1f);
        float max = Mathf.Clamp(Mathf.Max(range.x, range.y), 0.01f, 1f);

        return Random.Range(min, max);
    }

    void ResetArena()
    {
        ApplyBlockSize();

        var envParams = Academy.Instance.EnvironmentParameters;

        float cheatRate = envParams.GetWithDefault(CheatRateParameter, defaultCheatRate);
        bool multiCheat = envParams.GetWithDefault(MultiCheatParameter, allowMultipleCheats ? 1f : 0f) > 0.5f;
        float axisDeg = randomiseSpawns ? Random.Range(0f, 360f) : 0f;

        ResetAgent(agentA, spawnA, axisDeg);
        ResetAgent(agentB, spawnB, axisDeg + 180f);

        AssignCheats(agentA, cheatRate, multiCheat);
        AssignCheats(agentB, cheatRate, multiCheat);

        roundOver = false;
    }

    void ResetAgent(ShooterAgent agent, Transform spawn, float axisDeg)
    {
        Rigidbody rb = agent.GetComponent<Rigidbody>();
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        Vector3 position;
        Quaternion rotation;

        if (radialSpawns)
        {
            Vector3 outward = Quaternion.Euler(0f, axisDeg, 0f) * Vector3.forward;

            position = transform.TransformPoint(outward * spawnRadius);
            rotation = Quaternion.LookRotation(transform.TransformDirection(-outward), Vector3.up);
        }
        else
        {
            spawn.GetPositionAndRotation(out position, out rotation);
        }

        if (randomiseSpawns)
        {
            position += new Vector3(
                Random.Range(-spawnPositionJitter, spawnPositionJitter),
                0f,
                Random.Range(-spawnPositionJitter, spawnPositionJitter));

            rotation *= Quaternion.Euler(0f, Random.Range(-spawnYawJitter, spawnYawJitter), 0f);
        }

        rb.position = position;
        rb.rotation = rotation;
        agent.transform.SetPositionAndRotation(position, rotation);

        agent.GetComponent<Health>().ResetHealth();
        agent.ResetForNewRound();
    }

    public ShooterAgent GetOpponent(ShooterAgent agent)
    {
        return agent == agentA ? agentB : agentA;
    }

    public Vector2 NormalisedPosition(Vector3 worldPosition)
    {
        Vector3 local = transform.InverseTransformPoint(worldPosition);
        return new Vector2(
            Mathf.Clamp(local.x / (arenaSize.x * 0.5f), -1f, 1f),
            Mathf.Clamp(local.z / (arenaSize.y * 0.5f), -1f, 1f));
    }

    void OnDrawGizmosSelected()
    {
        Gizmos.matrix = transform.localToWorldMatrix;

        Gizmos.color = Color.cyan;
        Gizmos.DrawWireCube(Vector3.zero, new Vector3(arenaSize.x, 0.1f, arenaSize.y));

        if (!radialSpawns) return;

        Gizmos.color = Color.green;

        const int segments = 48;
        Vector3 previous = new Vector3(0f, 0f, spawnRadius);

        for (int i = 1; i <= segments; i++)
        {
            Vector3 next = Quaternion.Euler(0f, i * 360f / segments, 0f) * new Vector3(0f, 0f, spawnRadius);
            Gizmos.DrawLine(previous, next);
            previous = next;
        }

        Gizmos.DrawLine(new Vector3(0f, 0f, spawnRadius), new Vector3(0f, 0f, -spawnRadius));
    }
}
