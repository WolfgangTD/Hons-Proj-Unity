using System.Globalization;
using System.Text;
using UnityEngine;

/// <summary>
/// Per-agent recorder. Turns one decision into one row of server-observable behaviour,
/// and one episode into one labelled summary row.
///
/// Everything written to steps.csv is measurable by an authoritative server that trusts
/// no client: both players' transforms, both velocities, line of sight from the world
/// geometry, and the shots that arrive. The cheat severities are never written there,
/// only to episodes.csv, so a sequence model has to learn behaviour rather than read
/// the answer off the feature vector.
/// </summary>
public class AgentTelemetry : MonoBehaviour
{
    const string StepHeader =
        "session,episode_id,arena_id,agent_id,decision,t_s," +
        "pos_x,pos_z,yaw_deg,turn_deg,vel_x,vel_z,speed," +
        "enemy_pos_x,enemy_pos_z,dist,aim_error_deg,aim_error_delta_deg," +
        "has_los,los_state_age_s,target_in_cone,cone_dwell_s," +
        "weapon_ready,shot,shot_result,damage_dealt," +
        "health,enemy_health,enemy_yaw_rel_deg,enemy_speed";

    const string EpisodeHeader =
        "session,episode_id,arena_id,agent_id,ended_utc,duration_s,decisions,outcome," +
        "shots_fired,shots_hit,accuracy,damage_dealt,damage_taken," +
        "mean_abs_aim_error_deg,mean_abs_aim_error_los_deg,cumulative_reward,block_size," +
        "label_cheating,label_type,label_wallhack,label_aim_snap,label_trigger_bot," +
        "gt_wallhack_reveals,gt_aimsnap_adjustments,gt_triggerbot_shots";

    static readonly StringBuilder Row = new StringBuilder(512);

    [Tooltip("Stable name for this seat in the arena. Defaults to the GameObject name.")]
    public string agentId = "";

    [Tooltip("Log every physics tick instead of every policy decision. With Take Actions " +
             "Between Decisions on, that is Decision Period times as many rows (50Hz here), " +
             "and is closer to what a real server tickrate would give a detector.")]
    public bool logEveryPhysicsTick = false;

    ShooterAgent agent;
    ArenaManager arena;
    Weapon weapon;
    Health health;
    PlayerMotor motor;

    ShooterAgent enemy;
    Health enemyHealth;
    PlayerMotor enemyMotor;

    bool active;
    int episodeId;
    int arenaId;
    int decision;
    float episodeStart;
    float blockSize;

    bool decisionPending;

    // Shots resolve on any tick, so they are banked here and reported by the next row
    // rather than being lost on the ticks that are not logged.
    int pendingShots;
    int pendingHits;
    float pendingDamage;

    // Continuous state carried between decisions.
    float prevAimError;
    bool hasPrevAimError;
    bool hadLos;
    float losChangedAt;
    bool wasInCone;
    float coneEnteredAt;

    // Episode accumulators.
    int shotsFired;
    int shotsHit;
    float damageDealt;
    float startHealth;
    float aimErrorSum;
    int aimErrorSamples;
    float aimErrorLosSum;
    int aimErrorLosSamples;

    // Ground truth, written only to episodes.csv.
    int wallhackReveals;
    int aimSnapAdjustments;
    int triggerBotShots;

    void Awake()
    {
        agent = GetComponent<ShooterAgent>();
        weapon = GetComponent<Weapon>();
        health = GetComponent<Health>();
        motor = GetComponent<PlayerMotor>();

        if (string.IsNullOrEmpty(agentId)) agentId = name;

        // The id is written unquoted, so a comma in a GameObject name would shift every
        // column after it.
        agentId = agentId.Replace(',', '_');
    }

    /// <summary>Called by ArenaManager once the spawn and the cheat draw are both settled.</summary>
    public void BeginEpisode(int newEpisodeId, int newArenaId, float currentBlockSize)
    {
        if (!TelemetryWriter.Enabled) return;

        // Wiring is resolved here rather than in Awake, because ShooterAgent.Initialize
        // is what fills in the enemy reference.
        arena = agent.arena != null ? agent.arena : GetComponentInParent<ArenaManager>();
        enemy = agent.enemy;

        if (enemy == null || arena == null) return;

        enemyHealth = enemy.GetComponent<Health>();
        enemyMotor = enemy.GetComponent<PlayerMotor>();

        episodeId = newEpisodeId;
        arenaId = newArenaId;
        blockSize = currentBlockSize;

        decision = 0;
        episodeStart = Time.time;

        decisionPending = false;
        pendingShots = 0;
        pendingHits = 0;
        pendingDamage = 0f;

        hasPrevAimError = false;
        hadLos = false;
        losChangedAt = Time.time;
        wasInCone = false;
        coneEnteredAt = Time.time;

        shotsFired = 0;
        shotsHit = 0;
        damageDealt = 0f;
        startHealth = health.CurrentHealth;
        aimErrorSum = 0f;
        aimErrorSamples = 0;
        aimErrorLosSum = 0f;
        aimErrorLosSamples = 0;

        wallhackReveals = 0;
        aimSnapAdjustments = 0;
        triggerBotShots = 0;

        active = true;
    }

    /// <summary>
    /// Called from ShooterAgent.OnActionReceived once the shot has resolved but before its
    /// damage is applied.
    /// </summary>
    public void RecordDecision(float turnAppliedDeg, float weaponReady, bool shotAttempted, ShotResult result)
    {
        if (!active) return;

        // Banked before the gate below. A 0.2s cooldown against a 0.1s decision period means
        // shots land on ticks that carry no row, and dropping those would understate every
        // shooter in the capture.
        if (shotAttempted && result != ShotResult.NotReady)
        {
            shotsFired++;
            pendingShots++;

            if (result == ShotResult.Hit)
            {
                shotsHit++;
                pendingHits++;
                damageDealt += weapon.damage;
                pendingDamage += weapon.damage;
            }
        }

        bool decisionStep = decisionPending;
        decisionPending = false;

        // Everything past here costs a raycast and an overlap sphere, so it runs only on
        // rows that are actually written.
        if (!logEveryPhysicsTick && !decisionStep) return;

        float now = Time.time;

        Vector3 localPos = arena.transform.InverseTransformPoint(transform.position);
        Vector3 enemyLocalPos = arena.transform.InverseTransformPoint(enemy.transform.position);

        float yaw = Mathf.DeltaAngle(arena.transform.eulerAngles.y, transform.eulerAngles.y);
        float enemyYawRel = Mathf.DeltaAngle(transform.eulerAngles.y, enemy.transform.eulerAngles.y);

        Vector3 vel = arena.transform.InverseTransformDirection(motor.Velocity);

        Vector3 toEnemy = enemy.transform.position - transform.position;
        float dist = toEnemy.magnitude;
        toEnemy.y = 0f;

        float aimError = Vector3.SignedAngle(transform.forward, toEnemy, Vector3.up);
        float aimErrorDelta = hasPrevAimError ? Mathf.DeltaAngle(prevAimError, aimError) : 0f;
        prevAimError = aimError;
        hasPrevAimError = true;

        bool hasLos = weapon.HasLineOfSight(agent.aimPoint.position, enemy.aimPoint.position);
        if (hasLos != hadLos)
        {
            hadLos = hasLos;
            losChangedAt = now;
        }

        // Dwell time on target is the trigger-bot tell: an honest player needs reaction
        // time between the target entering the cone and the shot, a bot needs none.
        bool inCone = weapon.PeekTarget() != null;
        if (inCone != wasInCone)
        {
            wasInCone = inCone;
            coneEnteredAt = now;
        }

        aimErrorSum += Mathf.Abs(aimError);
        aimErrorSamples++;

        if (hasLos)
        {
            aimErrorLosSum += Mathf.Abs(aimError);
            aimErrorLosSamples++;
        }

        Row.Clear();
        Text(TelemetryWriter.SessionId);
        Int(episodeId);
        Int(arenaId);
        Text(agentId);
        Int(decision++);
        Num(now - episodeStart);

        Num(localPos.x);
        Num(localPos.z);
        Num(yaw);
        Num(turnAppliedDeg);
        Num(vel.x);
        Num(vel.z);
        Num(motor.Velocity.magnitude);

        Num(enemyLocalPos.x);
        Num(enemyLocalPos.z);
        Num(dist);
        Num(aimError);
        Num(aimErrorDelta);

        Int(hasLos ? 1 : 0);
        Num(now - losChangedAt);
        Int(inCone ? 1 : 0);
        Num(inCone ? now - coneEnteredAt : 0f);

        // The window since the previous row, not just this tick: 1 if a shot was taken,
        // Hit if any of them connected, and the damage they did between them.
        Num(weaponReady);
        Int(pendingShots > 0 ? 1 : 0);
        Int(pendingHits > 0 ? (int)ShotResult.Hit : pendingShots > 0 ? (int)ShotResult.Miss : -1);
        Num(pendingDamage);

        pendingShots = 0;
        pendingHits = 0;
        pendingDamage = 0f;

        Num(health.Fraction);
        Num(enemyHealth.Fraction);
        Num(enemyYawRel);
        Num(enemyMotor.Velocity.magnitude, last: true);

        TelemetryWriter.WriteStep(StepHeader, Row);
    }

    /// <summary>Called by ArenaManager after the terminal reward, before Agent.EndEpisode resets it.</summary>
    public void FinishEpisode(string outcome)
    {
        if (!active) return;
        active = false;

        // No decisions means the round ended before this agent ever acted; an empty
        // sequence would only add noise to the training set.
        if (decision == 0) return;

        Row.Clear();
        Text(TelemetryWriter.SessionId);
        Int(episodeId);
        Int(arenaId);
        Text(agentId);
        Text(System.DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture));
        Num(Time.time - episodeStart);
        Int(decision);
        Text(outcome);

        Int(shotsFired);
        Int(shotsHit);
        Num(shotsFired == 0 ? 0f : (float)shotsHit / shotsFired);
        Num(damageDealt);
        Num(startHealth - health.CurrentHealth);

        Num(aimErrorSamples == 0 ? 0f : aimErrorSum / aimErrorSamples);
        Num(aimErrorLosSamples == 0 ? -1f : aimErrorLosSum / aimErrorLosSamples);
        Num(agent.GetCumulativeReward());
        Num(blockSize);

        bool cheating = agent.wallhack > 0f || agent.aimSnap > 0f || agent.triggerBot > 0f;
        Int(cheating ? 1 : 0);
        Text(CheatType());
        Num(agent.wallhack);
        Num(agent.aimSnap);
        Num(agent.triggerBot);

        Int(wallhackReveals);
        Int(aimSnapAdjustments);
        Int(triggerBotShots, last: true);

        TelemetryWriter.WriteEpisode(EpisodeHeader, Row);
    }

    string CheatType()
    {
        int count = 0;
        if (agent.wallhack > 0f) count++;
        if (agent.aimSnap > 0f) count++;
        if (agent.triggerBot > 0f) count++;

        if (count == 0) return "none";
        if (count > 1) return "multi";

        if (agent.wallhack > 0f) return "wallhack";
        return agent.aimSnap > 0f ? "aimsnap" : "triggerbot";
    }

    /// <summary>
    /// Called from CollectObservations, which ML-Agents runs once per policy decision and
    /// never on the ticks in between. That makes it the honest marker for a decision step,
    /// without this having to guess at DecisionRequester's internal phase.
    /// </summary>
    public void NoteDecision()
    {
        decisionPending = true;
    }

    // Ground-truth counters. These say how often a cheat actually changed the outcome,
    // which is not the same as how severe it was configured to be.
    public void NoteWallhackReveal() { if (active) wallhackReveals++; }
    public void NoteAimSnapAdjustment() { if (active) aimSnapAdjustments++; }
    public void NoteTriggerBotShot() { if (active) triggerBotShots++; }

    static void Num(float value, bool last = false)
    {
        Row.Append(value.ToString("0.#####", CultureInfo.InvariantCulture));
        if (!last) Row.Append(',');
    }

    static void Int(int value, bool last = false)
    {
        Row.Append(value.ToString(CultureInfo.InvariantCulture));
        if (!last) Row.Append(',');
    }

    static void Text(string value, bool last = false)
    {
        Row.Append(value);
        if (!last) Row.Append(',');
    }
}
