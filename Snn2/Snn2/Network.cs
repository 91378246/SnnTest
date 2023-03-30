namespace Snn2;

public sealed class Network
{
    private static readonly int[] SIZE = new int[] { 2, 2, 2 };

    private List<Neuron>[] Layers = new List<Neuron>[3];

    public Network(Random rnd, int avgSpikeCountPerInputNeuron)
    {
        Console.ForegroundColor = ConsoleColor.Blue;
        Console.WriteLine($"Calculated optimal learning rate: {GetBestLearningRate()}");
        Console.ForegroundColor = ConsoleColor.Gray;

        // Neurons
        for (int layerI = 0; layerI < Layers.Length; layerI++)
        {
            for (int neuronI = 0; neuronI < SIZE[layerI]; neuronI++)
            {
                Layers[layerI].Add(new Neuron());
            }
        }

        // Synapses
        for (int layerI = 0; layerI < Layers.Length; layerI++)
        {
            for (int neuronI = 0; neuronI < SIZE[layerI]; neuronI++)
            {
                // Input
                if (layerI > 0)
                {
                    for (int neuronPreI = 0; neuronPreI < SIZE[layerI - 1]; neuronPreI++)
                    {
                        for (int synI = 0; synI < Parameters.SYN_PER_NEURON; synI++)
                        {
                            Layers[layerI][neuronI].SynapsesPre.Add();
                        }
                    }
                }

                // Output
            }
        }

        for (int inputNeuronI = 0; inputNeuronI < InputNeurons.Length; inputNeuronI++)
        {
            for (int outputNeuronI = 0; outputNeuronI < OutputNeurons.Length; outputNeuronI++)
            {
                (double[] weights, double[] delays) = GetSynapseInitValsRnd();
                for (int synapseI = 0; synapseI < Parameters.SYN_PER_NEURON; synapseI++)
                {
                    Synapses[synI] = new Synapse()
                    {
                        Weight = weights[synapseI],
                        Delay = delays[synapseI],
                        NeuronPre = InputNeurons[inputNeuronI],
                        NeuronPost = OutputNeurons[outputNeuronI],
                    };

                    InputNeurons[inputNeuronI].SynapsesPost.Add(Synapses[synI]);
                    OutputNeurons[inputNeuronI].SynapsesPre.Add(Synapses[synI]);
                    synI++;
                }
            }
        }

        double GetBestLearningRate() =>
            Parameters.THRESHOLD / (Synapses.Length * avgSpikeCountPerInputNeuron * (Parameters.TAU_M - Parameters.TAU_S));

        (double[] weights, double[] delays) GetSynapseInitVals(int prevLayerSize)
        {
            double div = Parameters.MIN_DELAY / Parameters.DECAY_TIME;
            double alpha = div * Math.Exp(1 - div);

            double w_min = Parameters.DECAY_TIME / (Parameters.SYN_PER_NEURON * prevLayerSize * alpha);
            double w_max = Parameters.DECAY_TIME / (Parameters.SYN_PER_NEURON * prevLayerSize * alpha);
            double[] weights = Enumerable.Repeat(0.0, Parameters.SYN_PER_NEURON).Select(d => rnd.NextDouble() * (w_max - w_min) + w_min).ToArray();
            double[] delays = Enumerable.Range(0, Parameters.SYN_PER_NEURON).Select(i => Parameters.MIN_DELAY + (Parameters.MAX_DELAY - Parameters.MIN_DELAY) * ((double)i / (Parameters.SYN_PER_NEURON <= 1 ? 1 : Parameters.SYN_PER_NEURON - 1))).ToArray();

            return (weights, delays);
        }

        (double[] weights, double[] delays) GetSynapseInitValsRnd()
        {
            double[] weights = Enumerable.Repeat(0.0, Parameters.SYN_PER_NEURON).Select(d => GetRandomNumber(-0.01, 0.1)).ToArray();
            double[] delays = Enumerable.Range(1, Parameters.SYN_PER_NEURON).Select(Convert.ToDouble).ToArray();

            return (weights, delays);
        }

        double GetRandomNumber(double minimum, double maximum)
        {
            Random random = new Random();
            return random.NextDouble() * (maximum - minimum) + minimum;
        }
    }

    public void Reset()
    {
        foreach (Neuron neuron in InputNeurons)
        {
            neuron.Reset();
        }

        foreach (Neuron neuron in OutputNeurons)
        {
            neuron.Reset();
        }
    }

    public void Forward(List<double>[] input, bool breakAtFirstSpike = false)
    {
        // Input
        for (int i = 0; i < InputNeurons.Length; i++)
        {
            InputNeurons[i].Spikes.AddRange(input[i]);
        }

        // Output
        for (int i = 0; i < OutputNeurons.Length; i++)
        {
            // Time
            for (double t = 0; t < Parameters.INTERVAL_DURATION; t += Parameters.TIME_STEP)
            {
                if (OutputNeurons[i].UpdatePotential(t, Synapses.Where(s => s.NeuronPost == OutputNeurons[i]).ToList())
                    && breakAtFirstSpike)
                {
                    break;
                }
            }
        }
    }

    public int GetCurrentlyPredictedClass()
    {
        double firstSpikeTime = double.MaxValue;
        int prediction = -1;
        for (int i = 0; i < OutputNeurons.Length; i++)
        {
            if (OutputNeurons[i].Spikes.FirstOrDefault() < firstSpikeTime)
            {
                firstSpikeTime = OutputNeurons[i].Spikes.FirstOrDefault();
                prediction = i;
            }
        }

        return prediction;
    }

    public double CalculateError(double[] desiredSpikeTimes)
    {
        double error = 0;

        for (int i = 0; i < desiredSpikeTimes.Length; i++)
        {
            error += Math.Pow(OutputNeurons[i].Spikes.FirstOrDefault() - desiredSpikeTimes[i], 2);
        }

        return 0.5 * error;
    }

    public void Backward(double[] desiredSpikeTimes)
    {
        // Neurons
        for (int i = 0; i < OutputNeurons.Length; i++)
        {
            OutputNeurons[i].UpdateInputWeights(desiredSpikeTimes[i]);
        }
    }

    public void Report(int epoch)
    {
        string reportNeurons = "";
        string reportSynapses = "";

        for (int i = 0; i < InputNeurons.Length; i++)
        {
            reportNeurons += $"Neuron[L0N{i}] spikes (first|total): {InputNeurons[i].Spikes.FirstOrDefault(-1)}|{InputNeurons[i].Spikes.Count}\n";
            reportNeurons += "Spike graph: ";
            for (double t = 0; t < Parameters.INTERVAL_DURATION; t += Parameters.TIME_STEP)
            {
                // Fix floating point error
                t = Math.Round(t, 1);
                reportNeurons += InputNeurons[i].Spikes.Any(ts => Math.Round(ts, 1) == t) ? "|" : ".";
            }
            reportNeurons += "\n";

            // Post-Neurons
            for (int j = 0; j < OutputNeurons.Length; j++)
            {
                // Synapses
                for (int s = 0; s < Parameters.SYN_PER_NEURON; s++)
                {
                    reportSynapses += $"Synapse {s} between [L0N{i}] and [L1N{j}] weight: {Synapses.Where(s => s.NeuronPre == InputNeurons[i] && s.NeuronPost == OutputNeurons[j]).First().Weight}\n";
                }
            }

            reportNeurons += "------------------------------------------------------------\n";
        }

        for (int i = 0; i < OutputNeurons.Length; i++)
        {
            reportNeurons += $"Neuron[L1N{i}] spikes (first|total): {InputNeurons[i].Spikes.FirstOrDefault(-1)}|{InputNeurons[i].Spikes.Count}\n";
            reportNeurons += "Spike graph: ";
            for (double t = 0; t < Parameters.INTERVAL_DURATION; t += Parameters.TIME_STEP)
            {
                // Fix floating point error
                t = Math.Round(t, 1);
                reportNeurons += InputNeurons[i].Spikes.Any(ts => Math.Round(ts, 1) == t) ? "|" : ".";
            }
            reportNeurons += "\n";
        }

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"\n##### START EPOCH {epoch} REPORT #####");
        Console.ForegroundColor = ConsoleColor.Gray;
        Console.WriteLine("NEURONS");
        Console.WriteLine(reportNeurons);
        Console.WriteLine("SYNAPSES");
        Console.WriteLine(reportSynapses);
        Console.WriteLine($"#####  END EPOCH {epoch} REPORT #####\n");
    }
}
