from simulator.simulator import EpisodeSimulator, SimulatorStats
from simulator.models import TaskType, FailureMode
from scoring.scorer import EpisodeScoringEngine

sim    = EpisodeSimulator(seed=42)
engine = EpisodeScoringEngine()
stats  = SimulatorStats()

print("Generating and scoring 100 episodes...\n")

results = {"CERTIFIED": 0, "BORDERLINE": 0, "REJECTED": 0}
detected = 0
total_with_failure = 0

for i in range(100):
    episode = sim.generate_episode()
    stats.record(episode)
    result  = engine.score(episode.to_json())
    results[result.routing_decision] += 1

    if episode.injected_failure.value != "none":
        total_with_failure += 1
        if result.routing_decision in ("BORDERLINE", "REJECTED"):
            detected += 1

    if i < 5:
        print(f"Episode {i+1}: {episode.task_type.value:15s} | "
              f"failure={episode.injected_failure.value:20s} | "
              f"score={result.composite_score:5.1f} | "
              f"{result.routing_decision}")

print(f"\n--- 100 Episode Summary ---")
print(f"CERTIFIED:  {results['CERTIFIED']}")
print(f"BORDERLINE: {results['BORDERLINE']}")
print(f"REJECTED:   {results['REJECTED']}")
if total_with_failure:
    print(f"\nFailure detection rate: {detected}/{total_with_failure} = {detected/total_with_failure*100:.1f}%")
print(f"\nSimulator stats: {stats.summary()}")