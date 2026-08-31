using UnityEngine;

/// <summary>
/// Editor-only training setup that must run before any scene object exists.
/// </summary>
public static class TrainingManager
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    static void EnableBackgroundExecution()
    {
        if (Application.isEditor)
        {
            Application.runInBackground = true;
        }
    }
}
