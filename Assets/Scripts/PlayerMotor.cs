using UnityEngine;

public class PlayerMotor : MonoBehaviour
{
    public float moveSpeed = 5f;

    Rigidbody rb;
    Vector3 lastPosition;

    public float MaxSpeed => moveSpeed;
    public Vector3 Velocity { get; private set; }

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        lastPosition = transform.position;
    }

    void FixedUpdate()
    {
        Vector3 position = transform.position;
        Velocity = (position - lastPosition) / Time.fixedDeltaTime;
        lastPosition = position;
    }

    public void ResetVelocityTracking()
    {
        lastPosition = transform.position;
        Velocity = Vector3.zero;
    }

    public void Move(float x, float z)
    {
        Vector3 dir = Vector3.ClampMagnitude(new Vector3(x, 0f, z), 1f);

        rb.MovePosition(rb.position + transform.TransformDirection(dir) * moveSpeed * Time.fixedDeltaTime);
    }

    public void RotateBy(float degrees)
    {
        if (Mathf.Approximately(degrees, 0f)) return;

        rb.MoveRotation(rb.rotation * Quaternion.Euler(0f, degrees, 0f));
    }
}
