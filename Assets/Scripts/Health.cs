using UnityEngine;

public class Health : MonoBehaviour
{
    public float maxHealth = 100f;

    public float CurrentHealth { get; private set; }

    ShooterAgent agent;

    public float Fraction => Mathf.Clamp01(CurrentHealth / maxHealth);

    void Awake()
    {
        agent = GetComponent<ShooterAgent>();
        CurrentHealth = maxHealth;
    }

    public void ResetHealth()
    {
        CurrentHealth = maxHealth;
    }
    public void TakeDamage(float damage)
    {
        if (CurrentHealth <= 0f) return;

        CurrentHealth = Mathf.Max(0f, CurrentHealth - damage);

        if (CurrentHealth <= 0f && agent != null && agent.arena != null)
            agent.arena.ReportDeath(agent);
    }
}
