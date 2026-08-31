using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

[RequireComponent(typeof(PlayerMotor))]
[RequireComponent(typeof(Weapon))]
[RequireComponent(typeof(Health))]
[RequireComponent(typeof(Rigidbody))]
public class ShooterAgent : Agent
{
    public const int ObservationSize = 21;

    [Header("Wiring")]
    public ArenaManager arena;
    public ShooterAgent enemy;
    public Transform aimPoint;

    [Header("Aim")]
    public float[] turnAnglesDeg = { -18f, -6f, -2f, 0f, 2f, 6f, 18f };

    [Header("Shot rewards")]
    public float hitRewardMax = 1f;
    public float missPenaltyMax = 1f;

    [Header("Cheats")]
    public bool manualCheatOverride = false;

    [Range(0f, 1f)]
    public float wallhack = 0f;

    [Range(0f, 1f)]
    public float aimSnap = 0f;

    [Range(0f, 1f)]
    public float triggerBot = 0f;

    [Header("Debug")]
    public float aimLineLength = 50f;
    public bool drawAimGizmo = true;
    public bool drawCheatIndicator = true;

    static readonly Color WallhackColour = new Color(1f, 0.2f, 1f);
    static readonly Color AimSnapColour = new Color(0.2f, 1f, 1f);
    static readonly Color TriggerBotColour = new Color(1f, 0.6f, 0f);

    PlayerMotor motor;
    Weapon weapon;
    Health health;
    PlayerMotor enemyMotor;
    Health enemyHealth;

    float hitReward;
    float missPenalty;

    public int ShotsFired;
    public int ShotsHit;
    public float Accuracy => ShotsFired == 0 ? 0f : (float)ShotsHit / ShotsFired;
    public Vector3 AimDirection => transform.forward;
    public override void Initialize()
    {
        motor = GetComponent<PlayerMotor>();
        weapon = GetComponent<Weapon>();
        health = GetComponent<Health>();

        if (arena == null) arena = GetComponentInParent<ArenaManager>();
        if (aimPoint == null) aimPoint = transform;

        enemyMotor = enemy.GetComponent<PlayerMotor>();
        enemyHealth = enemy.GetComponent<Health>();

        if (MaxStep != 0)
            Debug.Log($"{name}: set the Agent's Max Step to 0, handled by arena manager");

        ComputeShotRewards();
    }

    void ComputeShotRewards()
    {
        int shotsToKill = weapon.ShotsToKill(enemyHealth.maxHealth);
        hitReward = hitRewardMax / shotsToKill;
        float episodeSeconds = arena.MaxEpisodeSteps * Time.fixedDeltaTime;

        int cooldownLimit = Mathf.FloorToInt(episodeSeconds / Mathf.Max(0.0001f, weapon.cooldown));

        var requester = GetComponent<DecisionRequester>();
        int decisionPeriod = requester != null ? requester.DecisionPeriod : 1;
        int decisionLimit = arena.MaxEpisodeSteps / Mathf.Max(1, decisionPeriod);

        int maxShots = Mathf.Max(1, Mathf.Min(cooldownLimit, decisionLimit));

        missPenalty = missPenaltyMax / maxShots;
    }

    public void ResetForNewRound()
    {
        weapon.ResetCooldown();
        motor.ResetVelocityTracking();
        ShotsFired = 0;
        ShotsHit = 0;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        //Self
        Vector2 normPos = arena.NormalisedPosition(transform.position);
        sensor.AddObservation(normPos.x);
        sensor.AddObservation(normPos.y);

        float yawRad = transform.eulerAngles.y * Mathf.Deg2Rad;
        sensor.AddObservation(Mathf.Sin(yawRad));
        sensor.AddObservation(Mathf.Cos(yawRad));

        sensor.AddObservation(health.Fraction);
        sensor.AddObservation(weapon.ReadyFraction);

        Vector3 localVel = transform.InverseTransformDirection(motor.Velocity) / motor.MaxSpeed;
        sensor.AddObservation(Mathf.Clamp(localVel.x, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(localVel.z, -1f, 1f));

        bool hasLos = weapon.HasLineOfSight(aimPoint.position, enemy.aimPoint.position);
        sensor.AddObservation(hasLos ? 1f : 0f);

        //Enemy
        //Everything below is expressed in this agents local frame, per decision
        bool canSee = hasLos || (wallhack > 0f && Random.value < wallhack);

        Vector3 distToEnemy = enemy.transform.position - transform.position;
        Vector3 localdistToEnemy = transform.InverseTransformDirection(distToEnemy);
        Vector3 localDir = localdistToEnemy.sqrMagnitude > 0.0001f ? localdistToEnemy.normalized : Vector3.zero;

        float rotToEnemy = Vector3.SignedAngle(transform.forward, new Vector3(distToEnemy.x, 0f, distToEnemy.z), Vector3.up) / 180f;

        Vector3 relVel = transform.InverseTransformDirection(enemyMotor.Velocity - motor.Velocity) / motor.MaxSpeed;

        float enemyRelYawRad = Mathf.DeltaAngle(transform.eulerAngles.y, enemy.transform.eulerAngles.y) * Mathf.Deg2Rad;

        sensor.AddObservation(canSee ? localDir.x : 0f);
        sensor.AddObservation(canSee ? localDir.z : 0f);
        sensor.AddObservation(canSee ? Mathf.Clamp01(distToEnemy.magnitude / arena.MaxDistance) : 0f);
        sensor.AddObservation(canSee ? rotToEnemy : 0f);
        sensor.AddObservation(canSee ? enemyHealth.Fraction : 0f);
        sensor.AddObservation(canSee ? Mathf.Clamp(relVel.x, -1f, 1f) : 0f);
        sensor.AddObservation(canSee ? Mathf.Clamp(relVel.z, -1f, 1f) : 0f);
        sensor.AddObservation(canSee ? Mathf.Sin(enemyRelYawRad) : 0f);
        sensor.AddObservation(canSee ? Mathf.Cos(enemyRelYawRad) : 0f);

        //cheats
        sensor.AddObservation(wallhack);
        sensor.AddObservation(aimSnap);
        sensor.AddObservation(triggerBot);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float moveX = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float moveZ = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        int turnIndex = Mathf.Clamp(actions.DiscreteActions[0], 0, turnAnglesDeg.Length - 1);
        float turnDegrees = turnAnglesDeg[turnIndex];

        if (aimSnap > 0f) turnDegrees = ApplyAimSnap(turnDegrees);

        bool shoot = actions.DiscreteActions[1] == 1;

        motor.Move(moveX, moveZ);
        motor.RotateBy(turnDegrees);

        if (!shoot && triggerBot > 0f) shoot = TriggerBotWantsToFire();

        if (!shoot) return;

        ShotResult result = weapon.TryFire(out Health target);
        switch (result)
        {
            case ShotResult.Hit:
                ShotsFired++;
                ShotsHit++;
                AddReward(hitReward);
                target.TakeDamage(weapon.damage);
                break;

            case ShotResult.Miss:
                ShotsFired++;
                AddReward(-missPenalty);
                break;
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuous = actionsOut.ContinuousActions;
        var discrete = actionsOut.DiscreteActions;

        continuous[0] = Input.GetAxisRaw("Horizontal");
        continuous[1] = Input.GetAxisRaw("Vertical");

        discrete[0] = NearestTurnIndex(Input.GetAxis("Mouse X") * 10f);
        discrete[1] = Input.GetMouseButton(0) ? 1 : 0;
    }

    float ApplyAimSnap(float policyDegs)
    {
        Vector3 distToEnemy = enemy.transform.position - transform.position;
        distToEnemy.y = 0f;

        if (distToEnemy.sqrMagnitude < 0.0001f) return policyDegs;

        float idealDegrees = Vector3.SignedAngle(transform.forward, distToEnemy, Vector3.up);

        return turnAnglesDeg[NearestTurnIndex(Mathf.Lerp(policyDegs, idealDegrees, aimSnap))];
    }

    bool TriggerBotWantsToFire()
    {
        if (Random.value >= triggerBot) return false;

        return weapon.CanFire && weapon.PeekTarget() != null;
    }

    public void ApplyCheatConfig(float wallhackSeverity, float aimSnapSeverity, float triggerBotSeverity)
    {
        if (manualCheatOverride) return;

        wallhack = wallhackSeverity;
        aimSnap = aimSnapSeverity;
        triggerBot = triggerBotSeverity;
    }

    int NearestTurnIndex(float desiredDegrees)
    {
        int best = 0;
        float bestDelta = float.MaxValue;

        for (int i = 0; i < turnAnglesDeg.Length; i++)
        {
            float delta = Mathf.Abs(turnAnglesDeg[i] - desiredDegrees);
            if (delta < bestDelta)
            {
                bestDelta = delta;
                best = i;
            }
        }

        return best;
    }

    void Update()
    {
        if (drawAimGizmo) DrawAimGizmo();
        if (drawCheatIndicator) DrawCheatIndicator();
    }

    void DrawAimGizmo()
    {
        if (aimPoint == null) return;

        Vector3 origin = aimPoint.position;
        Debug.DrawRay(origin, AimDirection * aimLineLength, Color.red);

        float half = weapon != null ? weapon.coneHalfAngle : 0f;
        if (half > 0f)
        {
            Debug.DrawRay(origin, Quaternion.Euler(0f, half, 0f) * AimDirection * aimLineLength, Color.yellow);
            Debug.DrawRay(origin, Quaternion.Euler(0f, -half, 0f) * AimDirection * aimLineLength, Color.yellow);
        }
    }
    public readonly struct ActiveCheat
    {
        public readonly string Label;
        public readonly Color Colour;
        public readonly float Severity;

        public ActiveCheat(string label, Color colour, float severity)
        {
            Label = label;
            Colour = colour;
            Severity = severity;
        }
    }

    static readonly ActiveCheat[] CheatBuffer = new ActiveCheat[3];

    public int GetActiveCheats(ActiveCheat[] buffer)
    {
        int count = 0;

        if (wallhack > 0f)   buffer[count++] = new ActiveCheat("WALLHACK", WallhackColour, wallhack);
        if (aimSnap > 0f)    buffer[count++] = new ActiveCheat("AIMSNAP", AimSnapColour, aimSnap);
        if (triggerBot > 0f) buffer[count++] = new ActiveCheat("TRIGGERBOT", TriggerBotColour, triggerBot);

        return count;
    }
    void DrawCheatIndicator()
    {
        int count = GetActiveCheats(CheatBuffer);
        if (count == 0) return;

        Vector3 baseline = transform.position + Vector3.up * 2.2f;

        for (int i = 0; i < count; i++)
        {
            Vector3 foot = baseline + Vector3.right * ((i - (count - 1) * 0.5f) * 0.35f);
            Debug.DrawLine(foot, foot + Vector3.up * (0.3f + 1.2f * CheatBuffer[i].Severity), CheatBuffer[i].Colour);
        }
    }

#if UNITY_EDITOR
    //Text readout above the bar. Editor-only, not for build.
    void OnDrawGizmos()
    {
        if (!drawCheatIndicator) return;

        int count = GetActiveCheats(CheatBuffer);
        if (count == 0) return;

        string text = CheatBuffer[0].Label + " " + CheatBuffer[0].Severity.ToString("0.00");
        for (int i = 1; i < count; i++)
            text += "  +  " + CheatBuffer[i].Label + " " + CheatBuffer[i].Severity.ToString("0.00");

        var style = new GUIStyle
        {
            // A combination gets white text: no single cheat colour would be honest about it.
            normal = { textColor = count == 1 ? CheatBuffer[0].Colour : Color.white },
            fontStyle = FontStyle.Bold
        };

        UnityEditor.Handles.Label(transform.position + Vector3.up * 3.9f, text, style);
    }
#endif
}
