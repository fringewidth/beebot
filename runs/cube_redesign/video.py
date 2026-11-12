"""Side-by-side video: the scripted tower demo (cube_swarm.DEMO_SCRIPT) with the flywheel cube
(switchable magnets, as built) and with the hinge cube (cube_hinge, 0.3 N m, chamfered edges)."""
import sys
from beebot.hardware import cube_swarm as CS
from beebot.hardware.cube_hinge import HingeSwarm, Drive

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
for name, make in (("flywheel", lambda: CS.Swarm(CS.DEMO_CELLS, release=True, seed=seed)),
                   ("hinge", lambda: HingeSwarm(CS.DEMO_CELLS, drive=Drive(), seed=seed))):
    swarm = make()
    film = CS.Film(swarm, f"runs/cube_redesign/video/{name}-s{seed}.mp4", slow=1.0, size=(960, 720))
    film.still(swarm, 1.0)
    rows = CS.play(swarm, CS.DEMO_SCRIPT, film=film)
    film.still(swarm, 2.0)
    film.close()
    reached = sum(r["outcome"] == "reached" for r in rows)
    print(name, f"{reached} of {len(CS.DEMO_SCRIPT)} scripted moves reached; attempts {len(rows)}; outcomes", [r["outcome"] for r in rows if r["outcome"] != "reached"], flush=True)
