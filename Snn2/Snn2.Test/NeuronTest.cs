namespace Snn2.Test
{
    public sealed class NeuronTest
    {
        [Test]
        public void SpikeResponseFunctionTest()
        {
            double[] results = new[]
            {
                Neuron.SpikeResponseFunction(Neuron.CalculateS(spikeTimePreNeuron: 1, spikeTimePostNeuron: 0, delay: 0)),  // 0
                Neuron.SpikeResponseFunction(Neuron.CalculateS(spikeTimePreNeuron: 10, spikeTimePostNeuron: 9, delay: 0)), // 1
                Neuron.SpikeResponseFunction(Neuron.CalculateS(spikeTimePreNeuron: 1, spikeTimePostNeuron: 1, delay: 0)),  // 2
                Neuron.SpikeResponseFunction(Neuron.CalculateS(spikeTimePreNeuron: 1, spikeTimePostNeuron: 2, delay: 0)),  // 3
                Neuron.SpikeResponseFunction(Neuron.CalculateS(spikeTimePreNeuron: 1, spikeTimePostNeuron: 3, delay: 0)),  // 4
                Neuron.SpikeResponseFunction(Neuron.CalculateS(spikeTimePreNeuron: 1, spikeTimePostNeuron: 4, delay: 1)),  // 5
            };

            Assert.Multiple(() =>
            {
                Assert.That(results[0], Is.EqualTo(0));
                Assert.That(results[1], Is.EqualTo(0));
                Assert.That(results[2], Is.EqualTo(0));
                Assert.That(results[4], Is.GreaterThan(results[3]));
                Assert.That(results[5], Is.EqualTo(results[4]));
            });
        }

        [Test]
        public void ExponentialDecayTest()
        {
            double[] results = new[]
            {
                Neuron.ExponentialDecay(Neuron.CalculateS(spikeTimePreNeuron: 1, spikeTimePostNeuron: 0, delay: 0)),  // 0
                Neuron.ExponentialDecay(Neuron.CalculateS(spikeTimePreNeuron: 1, spikeTimePostNeuron: 1, delay: 0)),  // 1
                Neuron.ExponentialDecay(Neuron.CalculateS(spikeTimePreNeuron: 1, spikeTimePostNeuron: 2, delay: 0)),  // 2
                Neuron.ExponentialDecay(Neuron.CalculateS(spikeTimePreNeuron: 1, spikeTimePostNeuron: 3, delay: 0)),  // 3
                Neuron.ExponentialDecay(Neuron.CalculateS(spikeTimePreNeuron: 1, spikeTimePostNeuron: 4, delay: 1)),  // 4
            };

            Assert.Multiple(() =>
            {
                Assert.That(results[0], Is.EqualTo(0));
                Assert.That(results[1], Is.EqualTo(-1));
                Assert.That(results[2], Is.LessThan(results[3]));
                Assert.That(results[4], Is.EqualTo(results[3]));
            });
        }

        [Test]
        public void UpdatePotentialTest()
        {
            Neuron preNeuron = new();
            List<Synapse> preSynapses = new()
            {
                new Synapse()
                {
                    Weight = 10,
                    Delay = 0,
                    NeuronPre = preNeuron,
                }
            };

            preNeuron.Spikes = new List<double>() { 3 };
            Neuron neuronPost0 = new();
            preSynapses[0].NeuronPost = neuronPost0;
            neuronPost0.UpdatePotential(
                t: 5,
                preSynapses: preSynapses);

            preNeuron.Spikes = new List<double>() { 4 };
            Neuron neuronPost1 = new();
            preSynapses[0].NeuronPost = neuronPost1;
            neuronPost1.UpdatePotential(
                t: 5,
                preSynapses: preSynapses);

            preNeuron.Spikes = new List<double>() { 5 };
            Neuron neuronPost2 = new();
            preSynapses[0].NeuronPost = neuronPost2;
            neuronPost2.UpdatePotential(
                t: 5,
                preSynapses: preSynapses);

            preNeuron.Spikes = new List<double>() { 4 };
            Neuron neuronPost3 = new();
            preSynapses[0].NeuronPost = neuronPost3;
            preSynapses[0].Weight = 15;
            neuronPost3.UpdatePotential(
                t: 5,
                preSynapses: preSynapses);

            Assert.Multiple(() =>
            {
                Assert.That(neuronPost0.Potential, Is.GreaterThanOrEqualTo(0));
                Assert.That(neuronPost1.Potential, Is.LessThan(neuronPost0.Potential));
                Assert.That(neuronPost2.Potential, Is.LessThanOrEqualTo(0));
                Assert.That(neuronPost3.Potential, Is.GreaterThan(neuronPost1.Potential));
            });
        }
    }
}