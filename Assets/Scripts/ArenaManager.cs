using System.Globalization;
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

    static int nextArenaId;
    int arenaId;
    float currentBlockSize;

    public float MaxDistance => arenaSize.magnitude;

    public int MaxEpisodeSteps => maxEpisodeSteps;

    public int ArenaId => arenaId;

    // Resolved in one place so the draw and meta.json cannot disagree: the trainer's
    // curriculum when one is attached, the inspector fields otherwise.
    float CheatRate => Academy.Instance.EnvironmentParameters.GetWithDefault(CheatRateParameter, defaultCheatRate);
    bool MultiCheat => Academy.Instance.EnvironmentParameters.GetWithDefault(MultiCheatParameter, allowMultipleCheats ? 1f : 0f) > 0.5f;
    float BlockSize => Academy.Instance.EnvironmentParameters.GetWithDefault(BlockSizeParameter, defaultBlockSize);

    //statics survive a play session when domain reload is off, so ids would keep climbing
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    static void ResetArenaIds()
    {
        nextArenaId = 0;
    }

    void Awake()
    {
        // Stable within a process, so telemetry rows from cloned arenas stay separable.
        arenaId = nextArenaId++;
    }

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

        // Before EndEpisode, which clears the cumulative reward the summary row reports.
        winner.Telemetry?.FinishEpisode("win");
        loser.Telemetry?.FinishEpisode("loss");

        agentA.EndEpisode();
        agentB.EndEpisode();

        ResetArena();
    }

    void EndRoundDraw()
    {
        roundOver = true;

        agentA.AddReward(drawReward);
        agentB.AddReward(drawReward);

        agentA.Telemetry?.FinishEpisode("draw");
        agentB.Telemetry?.FinishEpisode("draw");

        //interupt instead of terminate
        agentA.EpisodeInterrupted();
        agentB.EpisodeInterrupted();

        ResetArena();
    }

    void ApplyBlockSize()
    {
        // Recorded even with no block assigned, so the telemetry still reports the lesson.
        float size = BlockSize;
        currentBlockSize = size;

        if (centreBlock == null) return;

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
        Vector2 bounds = SeverityBounds(range);

        return Random.Range(bounds.x, bounds.y);
    }

    /// <summary>The range RandomIn actually samples: ordered, and clamped into [0.01, 1].</summary>
    static Vector2 SeverityBounds(Vector2 range)
    {
        float min = Mathf.Clamp(Mathf.Min(range.x, range.y), 0.01f, 1f);
        float max = Mathf.Clamp(Mathf.Max(range.x, range.y), 0.01f, 1f);

        return new Vector2(min, max);
    }

    /// <summary>
    /// This arena's effective cheat settings as JSON fields, for meta.json. Ids are left
    /// out so that two arenas with the same settings produce the same string.
    /// </summary>
    public string CheatConfigJson()
    {
        Vector2 aimSnap = SeverityBounds(aimSnapRange);
        Vector2 triggerBot = SeverityBounds(triggerBotRange);

        return
            $"\"cheat_rate\": {Num(CheatRate)}, " +
            $"\"multi_cheat\": {(MultiCheat ? "true" : "false")}, " +
            $"\"aim_snap_range\": [{Num(aimSnap.x)}, {Num(aimSnap.y)}], " +
            $"\"trigger_bot_range\": [{Num(triggerBot.x)}, {Num(triggerBot.y)}], " +
            $"\"wallhack_severity\": {Num(WallhackSeverity)}, " +
            $"\"block_size\": {Num(BlockSize)}, " +
            $"\"max_episode_steps\": {maxEpisodeSteps.ToString(CultureInfo.InvariantCulture)}";
    }

    static string Num(float value) => value.ToString("R", CultureInfo.InvariantCulture);

    void ResetArena()
    {
        ApplyBlockSize();

        float cheatRate = CheatRate;
        bool multiCheat = MultiCheat;
        float axisDeg = randomiseSpawns ? Random.Range(0f, 360f) : 0f;

        ResetAgent(agentA, spawnA, axisDeg);
        ResetAgent(agentB, spawnB, axisDeg + 180f);

        AssignCheats(agentA, cheatRate, multiCheat);
        AssignCheats(agentB, cheatRate, multiCheat);

        BeginTelemetryEpisode();

        roundOver = false;
    }

    /// <summary>
    /// Opens the capture window only once the spawn and the cheat draw are both settled,
    /// so the episode's label row describes the episode that is about to be played.
    /// </summary>
    void BeginTelemetryEpisode()
    {
        if (!TelemetryWriter.Enabled) return;

        //both agents share the id, so a round can be reassembled from either side
        int episodeId = TelemetryWriter.NextEpisodeId();

        agentA.Telemetry?.BeginEpisode(episodeId, arenaId, currentBlockSize);
        agentB.Telemetry?.BeginEpisode(episodeId, arenaId, currentBlockSize);
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
