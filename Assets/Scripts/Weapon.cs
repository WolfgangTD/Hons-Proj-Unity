using UnityEngine;

public enum ShotResult
{
    NotReady,
    Miss,
    Hit
}

public class Weapon : MonoBehaviour
{
    [Header("WeaponStats")]
    public float damage = 20f;
    public float range = 50f;
    public float cooldown = 0.2f;

    [Header("Hit detection")]
    [Tooltip("Half-angle of the hit cone in degrees. A target inside the cone with clear lOS is hit")]
    [Range(0f, 30f)]
    public float coneHalfAngle = 4f;

    [Tooltip("Layers that can be shot (the agents).")]
    public LayerMask targetMask;

    [Tooltip("Layers that block line of sight (the walls).")]
    public LayerMask obstacleMask;

    float nextFire;

    ShooterAgent agent;

    static readonly Collider[] CandidateBuffer = new Collider[16];

    public float ReadyFraction =>
        cooldown <= 0f ? 1f : Mathf.Clamp01(1f - (nextFire - Time.time) / cooldown);

    public bool CanFire => Time.time >= nextFire;

    public int ShotsToKill(float targetMaxHealth)
    {
        return Mathf.Max(1, Mathf.CeilToInt(targetMaxHealth / Mathf.Max(0.0001f, damage)));
    }

    void Awake()
    {
        agent = GetComponent<ShooterAgent>();
    }

    public void ResetCooldown()
    {
        nextFire = 0f;
    }
    public ShotResult TryFire(out Health target)
    {
        target = null;

        if (!CanFire) return ShotResult.NotReady;

        nextFire = Time.time + cooldown;

        target = FindTargetInCone(agent.aimPoint.position, agent.AimDirection);

        return target == null ? ShotResult.Miss : ShotResult.Hit;
    }

    public Health PeekTarget()
    {
        return FindTargetInCone(agent.aimPoint.position, agent.AimDirection);
    }
    Health FindTargetInCone(Vector3 origin, Vector3 direction)
    {
        if (coneHalfAngle <= 0f)
        {
            if (Physics.Raycast(origin, direction, out RaycastHit hit, range, targetMask | obstacleMask, QueryTriggerInteraction.Ignore))
            {
                Health hp = hit.collider.GetComponentInParent<Health>();
                if (hp != null && hp.gameObject != gameObject) return hp;
            }
            return null;
        }

        int count = Physics.OverlapSphereNonAlloc(origin, range, CandidateBuffer, targetMask, QueryTriggerInteraction.Ignore);

        Health best = null;
        float bestAngle = float.MaxValue;

        for (int i = 0; i < count; i++)
        {
            Health hp = CandidateBuffer[i].GetComponentInParent<Health>();
            if (hp == null || hp.gameObject == gameObject) continue;

            Vector3 targetPoint = AimTargetPoint(hp);
            Vector3 toTarget = targetPoint - origin;
            toTarget.y = 0f;
            float angle = Vector3.Angle(direction, toTarget);

            if (angle > coneHalfAngle || angle >= bestAngle) continue;

            if (!HasLineOfSight(origin, targetPoint)) continue;

            best = hp;
            bestAngle = angle;
        }

        return best;
    }

    static Vector3 AimTargetPoint(Health target)
    {
        ShooterAgent targetAgent = target.GetComponent<ShooterAgent>();

        return targetAgent != null && targetAgent.aimPoint != null ? targetAgent.aimPoint.position : target.transform.position;
    }
    public bool HasLineOfSight(Vector3 origin, Vector3 targetPosition)
    {
        Vector3 delta = targetPosition - origin;
        float distance = delta.magnitude;

        if (distance > range) return false;

        return !Physics.Raycast(origin, delta / distance, distance, obstacleMask, QueryTriggerInteraction.Ignore);
    }
}
