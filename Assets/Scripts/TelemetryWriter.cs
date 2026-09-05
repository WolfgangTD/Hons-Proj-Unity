using System;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

/// <summary>
/// Owns the capture files for one Unity process and hands out episode ids.
///
/// Two streams are written, and the split is the whole point:
///
///   steps.csv    one row per agent per decision. Server-observable behaviour ONLY.
///                Nothing in here is unavailable to a real authoritative game server,
///                and no cheat label appears in it.
///
///   episodes.csv one row per agent per episode. Carries the ground-truth cheat
///                severities and the diagnostic counters, joined to steps.csv on
///                (session, episode_id, agent_id).
///
/// Keeping labels out of the feature stream is what stops the detector learning a
/// shortcut to the answer instead of learning behaviour.
/// </summary>
public static class TelemetryWriter
{
    const string DirectoryVariable = "SHOOTER_TELEMETRY_DIR";
    const string EnabledVariable = "SHOOTER_TELEMETRY";

    static bool initialised;
    static bool failed;

    static StreamWriter stepWriter;
    static StreamWriter episodeWriter;

    static bool stepHeaderWritten;
    static bool episodeHeaderWritten;

    static int nextEpisodeId;
    static int rowsSinceFlush;

    const int FlushInterval = 2000;

    public static string SessionId { get; private set; } = "";
    public static string SessionDirectory { get; private set; } = "";

    static bool? enabledCache;

    /// <summary>Set SHOOTER_TELEMETRY=0 to run training without paying the capture cost.</summary>
    public static bool Enabled
    {
        get
        {
            if (failed) return false;

            if (enabledCache == null)
            {
                string flag = Environment.GetEnvironmentVariable(EnabledVariable);
                enabledCache = flag != "0" && flag != "false";
            }

            return enabledCache.Value;
        }
    }

    /// <summary>
    /// Opening here rather than on the first row is what gives SessionId a value before
    /// any recorder starts stamping it into rows.
    /// </summary>
    public static int NextEpisodeId()
    {
        EnsureOpen();
        return ++nextEpisodeId;
    }

    public static void WriteStep(string header, StringBuilder row)
    {
        if (!EnsureOpen()) return;

        WriteRow(stepWriter, header, ref stepHeaderWritten, row);
    }

    public static void WriteEpisode(string header, StringBuilder row)
    {
        if (!EnsureOpen()) return;

        WriteRow(episodeWriter, header, ref episodeHeaderWritten, row);

        // Episode boundaries are the only safe place to be interrupted: both files end on a
        // complete row, so a killed training run still leaves a loadable capture.
        Flush();
    }

    static void WriteRow(StreamWriter writer, string header, ref bool headerWritten, StringBuilder row)
    {
        try
        {
            // The header goes down once, on the stream's first row, so the schema always
            // travels with the data it describes.
            if (!headerWritten)
            {
                headerWritten = true;
                writer.Write(header);
                writer.Write('\n');
            }

            writer.Write(row.ToString());
            writer.Write('\n');
        }
        catch (Exception e)
        {
            // A full disk partway through an overnight run should cost the capture, not
            // the training run it was riding along with.
            Debug.LogError($"Telemetry disabled after a write failure: {e.Message}");
            Close();
            return;
        }

        if (++rowsSinceFlush < FlushInterval) return;

        rowsSinceFlush = 0;
        Flush();
    }

    static void Flush()
    {
        try
        {
            stepWriter?.Flush();
            episodeWriter?.Flush();
        }
        catch (Exception e)
        {
            Debug.LogError($"Telemetry disabled after a flush failure: {e.Message}");
            Close();
        }
    }

    static bool EnsureOpen()
    {
        if (initialised) return !failed;
        initialised = true;

        if (!Enabled)
        {
            failed = true;
            return false;
        }

        try
        {
            string root = Environment.GetEnvironmentVariable(DirectoryVariable);
            if (string.IsNullOrEmpty(root))
                root = Path.Combine(Application.dataPath, "..", "Telemetry");

            // The pid keeps parallel ML-Agents workers (--num-envs) out of each other's files,
            // and the milliseconds stop two opens in the same second from resolving to one
            // directory and silently truncating the earlier capture.
            SessionId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture)
                        + "_" + System.Diagnostics.Process.GetCurrentProcess().Id;

            SessionDirectory = Path.GetFullPath(Path.Combine(root, "session_" + SessionId));
            Directory.CreateDirectory(SessionDirectory);

            stepWriter = new StreamWriter(
                new FileStream(Path.Combine(SessionDirectory, "steps.csv"), FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 16));

            episodeWriter = new StreamWriter(
                new FileStream(Path.Combine(SessionDirectory, "episodes.csv"), FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 16));

            WriteMeta();

            Application.quitting += Close;

            Debug.Log($"Telemetry capturing to {SessionDirectory}");
            return true;
        }
        catch (Exception e)
        {
            Debug.LogError($"Telemetry disabled, could not open capture files: {e.Message}");
            failed = true;
            return false;
        }
    }

    static void WriteMeta()
    {
        string meta =
            "{\n" +
            $"  \"session\": \"{SessionId}\",\n" +
            $"  \"started_utc\": \"{DateTime.UtcNow:o}\",\n" +
            $"  \"unity_version\": \"{Application.unityVersion}\",\n" +
            $"  \"fixed_delta_time\": {Time.fixedDeltaTime.ToString("R", CultureInfo.InvariantCulture)},\n" +
            $"  \"time_scale\": {Time.timeScale.ToString("R", CultureInfo.InvariantCulture)}\n" +
            "}\n";

        File.WriteAllText(Path.Combine(SessionDirectory, "meta.json"), meta);
    }

    public static void Close()
    {
        // Cleared before the flush, because Close is also the error path: a writer that
        // throws on the way out must not be left in place for the next row to retry.
        StreamWriter steps = stepWriter;
        StreamWriter episodes = episodeWriter;

        stepWriter = null;
        episodeWriter = null;

        failed = true;
        Application.quitting -= Close;

        try
        {
            steps?.Dispose();
            episodes?.Dispose();
        }
        catch (Exception e)
        {
            Debug.LogError($"Telemetry could not close cleanly: {e.Message}");
        }
    }

    /// <summary>
    /// Clears the previous session before anything in the scene can ask for an episode id.
    /// EnteredPlayMode is too late for this: the editor raises it after the first Awake and
    /// Start, so a reset there lands *after* the opening episode has already taken id 1 and
    /// hands the same id out a second time.
    /// </summary>
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    static void ResetForNewSession()
    {
        initialised = false;
        failed = false;
        enabledCache = null;
        nextEpisodeId = 0;
        stepHeaderWritten = false;
        episodeHeaderWritten = false;
    }

#if UNITY_EDITOR
    // Leaving play mode is not a quit, so Application.quitting never fires in the editor
    // and the last buffered rows would be lost.
    [UnityEditor.InitializeOnLoadMethod]
    static void CloseOnExitPlayMode()
    {
        UnityEditor.EditorApplication.playModeStateChanged += state =>
        {
            if (state == UnityEditor.PlayModeStateChange.ExitingPlayMode) Close();
        };
    }
#endif
}
