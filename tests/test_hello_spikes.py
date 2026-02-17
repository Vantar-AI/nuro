"""Integration test — full API flow matching hello_spikes.py."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spikingjelly")

import nuro


class TestHelloSpikes:
    def test_full_flow(self):
        # Create two populations
        input_pop = nuro.Population(size=100, dynamics="lif", params={"tau": 20e-3})
        output_pop = nuro.Population(size=10, dynamics="lif", params={"tau": 10e-3})

        # Connect them
        conn = nuro.Connection(
            source=input_pop,
            target=output_pop,
            pattern="dense",
            plasticity="stdp",
        )

        # Build and compile
        graph = nuro.Graph([input_pop, output_pop], [conn])
        runtime = nuro.compile(graph, target="gpu")

        # Run for 1 second
        runtime.run(duration=1.0)

        # Verify metrics
        assert "total_spikes" in runtime.metrics
        assert runtime.metrics["total_spikes"] > 0
        assert runtime.metrics["num_steps"] == 1000
        assert input_pop.id in runtime.metrics["spike_counts"]
        assert output_pop.id in runtime.metrics["spike_counts"]

    def test_three_layer_network(self):
        p1 = nuro.Population(size=100, dynamics="lif", params={"tau": 20e-3})
        p2 = nuro.Population(size=50, dynamics="lif", params={"tau": 15e-3})
        p3 = nuro.Population(size=10, dynamics="lif", params={"tau": 10e-3})

        c1 = nuro.Connection(source=p1, target=p2, pattern="dense")
        c2 = nuro.Connection(source=p2, target=p3, pattern="dense", plasticity="stdp")

        graph = nuro.Graph([p1, p2, p3], [c1, c2])
        runtime = nuro.compile(graph)
        runtime.run(duration=0.5)

        assert runtime.metrics["total_spikes"] > 0
