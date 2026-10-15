using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Unity.MLAgents;
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

    static StreamWriter activationWriter;
    static bool activationHeaderWritten;

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

    /// <summary>
    /// One row per moment a cheat actually fired. This is ground truth taken from the
    /// cheat's own code path, not re-derived from the feature stream, which is what makes
    /// it usable for checking whether a model attends to the right ticks.
    /// </summary>
    public static void WriteActivation(string header, StringBuilder row)
    {
        if (!EnsureOpen()) return;

        WriteRow(activationWriter, header, ref activationHeaderWritten, row);
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
            activationWriter?.Flush();
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

            activationWriter = new StreamWriter(
                new FileStream(Path.Combine(SessionDirectory, "activations.csv"), FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 16));

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

    /// <summary>
    /// The cheat settings are inspector fields toggled between runs, and the curriculum
    /// falls back to them silently when no trainer is attached, so a session that does not
    /// record them cannot say what produced it. Written per arena because nothing forces
    /// the arenas to agree. A snapshot at open: with a trainer attached, the curriculum can
    /// still move these mid-session, which trainer_connected flags.
    /// </summary>
    static void WriteMeta()
    {
        // Every ArenaManager has run Awake by now: the first open comes from an arena's Start.
        ArenaManager[] arenas = UnityEngine.Object.FindObjectsByType<ArenaManager>(FindObjectsSortMode.None);
        Array.Sort(arenas, (a, b) => a.ArenaId.CompareTo(b.ArenaId));

        var configs = new string[arenas.Length];
        var counts = new Dictionary<string, int>();
        string common = null;

        for (int i = 0; i < arenas.Length; i++)
        {
            configs[i] = arenas[i].CheatConfigJson();
            counts.TryGetValue(configs[i], out int n);
            counts[configs[i]] = ++n;

            if (common == null || n > counts[common]) common = configs[i];
        }

        bool consistent = counts.Count <= 1;

        var meta = new StringBuilder();
        meta.Append("{\n");
        meta.Append($"  \"session\": \"{SessionId}\",\n");
        meta.Append($"  \"started_utc\": \"{DateTime.UtcNow:o}\",\n");
        meta.Append($"  \"unity_version\": \"{Application.unityVersion}\",\n");
        meta.Append($"  \"fixed_delta_time\": {Time.fixedDeltaTime.ToString("R", CultureInfo.InvariantCulture)},\n");
        meta.Append($"  \"time_scale\": {Time.timeScale.ToString("R", CultureInfo.InvariantCulture)},\n");
        meta.Append($"  \"trainer_connected\": {Bool(Academy.Instance.IsCommunicatorOn)},\n");
        meta.Append($"  \"cheat_config_consistent\": {Bool(consistent)},\n");
        meta.Append("  \"arenas\": [");

        for (int i = 0; i < arenas.Length; i++)
        {
            ArenaManager arena = arenas[i];

            meta.Append(i == 0 ? "\n" : ",\n");
            meta.Append($"    {{\"arena_id\": {arena.ArenaId.ToString(CultureInfo.InvariantCulture)}, ");
            meta.Append($"\"object\": {Quote(arena.name)}, ");
            meta.Append($"\"agents\": [{Quote(AgentName(arena.agentA))}, {Quote(AgentName(arena.agentB))}], ");
            meta.Append(configs[i]);
            meta.Append('}');
        }

        meta.Append(arenas.Length == 0 ? "]\n" : "\n  ]\n");
        meta.Append("}\n");

        File.WriteAllText(Path.Combine(SessionDirectory, "meta.json"), meta.ToString());

        if (consistent) return;

        // A capture where one arena draws differently is confounded by arena, and the
        // per-episode labels cannot show it. Name the odd ones out so they can be fixed.
        var odd = new StringBuilder();
        for (int i = 0; i < arenas.Length; i++)
        {
            if (configs[i] != common)
                odd.Append($"\n  arena {arenas[i].ArenaId} ({arenas[i].name}): {configs[i]}");
        }

        Debug.LogWarning($"Arenas disagree on cheat settings, so this capture is confounded by arena. Most use:\n  {common}\nThese differ:{odd}");
    }

    static string AgentName(ShooterAgent agent)
    {
        if (agent == null) return "";

        AgentTelemetry recorder = agent.Telemetry;
        return recorder != null ? recorder.agentId : agent.name;
    }

    static string Bool(bool value) => value ? "true" : "false";

    static string Quote(string value) => "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";

    public static void Close()
    {
        // Cleared before the flush, because Close is also the error path: a writer that
        // throws on the way out must not be left in place for the next row to retry.
        StreamWriter steps = stepWriter;
        StreamWriter episodes = episodeWriter;
        StreamWriter activations = activationWriter;

        stepWriter = null;
        episodeWriter = null;
        activationWriter = null;

        failed = true;
        Application.quitting -= Close;

        try
        {
            steps?.Dispose();
            episodes?.Dispose();
            activations?.Dispose();
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
        activationHeaderWritten = false;
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
