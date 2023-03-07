using SnnTest;

const int SEED = 1;
const int EPOCHS = 50;
const int SAMPLE_COUNT = 1000;
const int TEST_COUNT = 100;

Random rnd = new(SEED);
(double[][] samples, bool[] labels) = GenerateXORData(SAMPLE_COUNT, rnd);
List<double[]> desiredSpikeTimes = new();
for (int i = 0; i < SAMPLE_COUNT; i++)
{
    desiredSpikeTimes.Add(labels[i] ? new double[] { 10, 6 } : new double[] { 6, 10 }); // Magic numbers ???
}
Network network = new(new int[] { 2, 2 }, rnd, 1);

Console.WriteLine($"Fitting XOR for {EPOCHS} epochs");

double lowestError = double.MaxValue;
double avgError = 0;
for (int epoch = 0; epoch < EPOCHS; epoch++)
{
    double error = 0;
    for (int i = 0; i < SAMPLE_COUNT; i++)
    {
        Shuffle(samples, labels, rnd);
        network.Reset();
        network.Forward(samples[i]);
        error += network.CalculateError(desiredSpikeTimes[i]);
        network.Backward(desiredSpikeTimes[i]);
    }

    // Console.Clear();
    error /= SAMPLE_COUNT;
    avgError += error;
    if (error < lowestError)
    {
        lowestError = error;
    }

    if (epoch % 10 == 0)
    {
        Console.WriteLine($"Epoch {epoch} error: {error}");
        network.Report(epoch);
    }
}

network.Report(EPOCHS - 1);
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"\n##### DONE #####");
Console.ForegroundColor = ConsoleColor.Gray;
Console.WriteLine($"Average error: {avgError /= EPOCHS}");
Console.WriteLine($"Lowest error: {lowestError}\n");

Console.WriteLine($"Running {TEST_COUNT} tests ...");
ConfusionMatrix cm = new();
(samples, labels) = GenerateXORData(TEST_COUNT, rnd);
for (int i = 0; i < TEST_COUNT; i++)
{
    Shuffle(samples, labels, rnd);
    network.Reset();
    network.Forward(samples[i]);

    bool prediction = Convert.ToBoolean(network.GetCurrentlyPredictedClass());
    if (prediction)
    {
        if (labels[i])
        {
            cm.TruePositives++;
        }
        else
        {
            cm.FalsePositives++;
        }
    }
    else
    {
        if (labels[i])
        {
            cm.FalseNegatives++;
        }
        else
        {
            cm.TrueNegatives++;
        }
    }

}
Console.WriteLine(cm.ToString());
Console.ReadLine();

static (double[][], bool[]) GenerateXORData(int n, Random rnd, bool shuffle = true, double sigma = 0.075)
{
    int n_half = n / 2;
    double[][] x = new double[n][];

    for (int i = 0; i < n_half; i++)
    {
        x[i] = new double[2];
        x[i][0] = NextGaussian(rnd, 0.25, sigma);
        x[i][1] = NextGaussian(rnd, 0.25, sigma);
    }
    for (int i = 0; i < n_half; i++)
    {
        for (int j = 0; j < x[i].Length; j++)
        {
            x[i][j] = x[i][j] < 0 ? -x[i][j] : x[i][j];
        }
    }

    for (int i = n_half; i < n; i++)
    {
        x[i] = new double[2];
        x[i][0] = NextGaussian(rnd, 0.75, sigma);
        x[i][1] = NextGaussian(rnd, 0.75, sigma);
    }
    for (int i = n_half; i < n; i++)
    {
        for (int j = 0; j < x[i].Length; j++)
        {
            x[i][j] = x[i][j] > 1 ? -x[i][j] % 1 : x[i][j];
        }
    }

    bool[] y = new bool[n];
    Array.Fill(y, true, n_half, n - n_half);

    if (shuffle)
    {
        Shuffle(x, y, rnd);
    }

    return (x, y);
}

static double NextGaussian(Random rng, double mean, double stdDev)
{
    double u1 = rng.NextDouble();
    double u2 = rng.NextDouble();
    double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    return mean + z0 * stdDev;
}

static void Shuffle(double[][] values, bool[] labels, Random rnd)
{
    // Get the dimensions.
    int num_rows = values.Length;
    int num_cols = values[0].Length;
    int num_cells = num_rows * num_cols;

    // Randomize the array.
    for (int i = 0; i < num_cells - 1; i++)
    {
        // Pick a random cell between i and the end of the array.
        int j = rnd.Next(i, num_cells);

        // Convert to row/column indexes.
        int row_i = i / num_cols;
        int col_i = i % num_cols;
        int row_j = j / num_cols;
        int col_j = j % num_cols;

        // Swap cells i and j.
        (values[row_j][col_j], values[row_i][col_i]) = (values[row_i][col_i], values[row_j][col_j]);

        // Swap the labels the same way
        (labels[row_j], labels[row_i]) = (labels[row_i], labels[row_j]);
    }
}

sealed class Network
{
    private const double THRESHOLD = 1;
    private const double MIN_DELAY = 1;
    private const double MAX_DELAY = 8;
    private const double DECAY_TIME = 7;
    private const double INTERVAL_DURATION = 25;
    private const double TIME_STEP = 0.1;
    private const double LEARNING_RATE = 0.0125;
    private const int SYN_PER_NEURON = 5;

    private List<List<Neuron>> Neurons { get; } = new();
    private int[] Dimensions { get; }

    public Network(int[] dimensions, Random rnd, int avgSpikeCountPerInputNeuron)
    {
        Console.ForegroundColor = ConsoleColor.Blue;
        Console.WriteLine($"Calculated optimal learning rate: {GetBestLearningRate()}");
        Console.ForegroundColor = ConsoleColor.Gray;

        Dimensions = dimensions;

        // Layers
        for (int l = 0; l < Dimensions.Length; l++)
        {
            Neurons.Add(new List<Neuron>());

            // Neurons
            for (int i = 0; i < Dimensions[l]; i++)
            {
                Neurons[^1].Add(new Neuron());
            }
        }

        // Layers
        double alpha = CalculateAlpha();
        for (int l = 0; l < Dimensions.Length - 1; l++)
        {
            // Pre-Neurons
            for (int i = 0; i < Dimensions[l]; i++)
            {
                (double[] weights, double[] delays) = GetSynapseInitVals(Dimensions[l]);

                // Post Neurons
                for (int j = 0; j < Dimensions[l + 1]; j++)
                {
                    // Synapses
                    for (int s = 0; s < SYN_PER_NEURON; s++)
                    {
                        Synapse syn = new(Neurons[l][i], Neurons[l + 1][j], weights[s], delays[s]);
                        Neurons[l][i].SynapsesOut.Add(syn);
                        Neurons[l + 1][j].SynapsesIn.Add(syn);
                    }
                }
            }
        }

        double GetBestLearningRate() =>
            THRESHOLD / (dimensions[^2] * SYN_PER_NEURON * avgSpikeCountPerInputNeuron * (Neuron.TAU_M - Neuron.TAU_S));

        static double CalculateAlpha()
        {
            double div = MIN_DELAY / DECAY_TIME;
            return div * Math.Exp(1 - div);
        }

        (double[] weights, double[] delays) GetSynapseInitVals(int prevLayerSize)
        {
            double w_min = DECAY_TIME / (SYN_PER_NEURON * prevLayerSize * alpha);
            double w_max = DECAY_TIME / (SYN_PER_NEURON * prevLayerSize * alpha);
            double[] weights = Enumerable.Repeat(0.0, SYN_PER_NEURON).Select(d => rnd.NextDouble() * (w_max - w_min) + w_min).ToArray();
            double[] delays = Enumerable.Range(0, SYN_PER_NEURON).Select(i => MIN_DELAY + (MAX_DELAY - MIN_DELAY) * ((double)i / (SYN_PER_NEURON <= 1 ? 1 : SYN_PER_NEURON - 1))).ToArray();

            return (weights, delays);
        }
    }

    public void Reset()
    {
        // Layers
        foreach (List<Neuron> layer in Neurons)
        {
            // Neurons
            foreach (Neuron neuron in layer)
            {
                neuron.Reset();
            }
        }
    }

    public void Forward(double[] input, bool breakAtFirstSpike = true, int roundTimeStepToDigits = 1)
    {
        // Input
        for (int i = 0; i < Dimensions[0]; i++)
        {
            Neurons[0][i].EncodeInput(input[i], MAX_DELAY);
        }

        // Layers
        for (int l = 1; l < Dimensions.Length; l++)
        {
            // Neurons
            for (int i = 0; i < Dimensions[l]; i++)
            {
                // Time
                for (double t = 0; t < INTERVAL_DURATION; t += TIME_STEP)
                {
                    // Fix floating point error
                    t = Math.Round(t, roundTimeStepToDigits);

                    if (Neurons[l][i].CalculatePotential(t, THRESHOLD, DECAY_TIME) && breakAtFirstSpike)
                    {
                        break;
                    }
                }
            }
        }
    }

    public int GetCurrentlyPredictedClass()
    {
        double firstSpikeTime = double.MaxValue;
        int prediction = -1;
        for (int i = 0; i < Dimensions[^1]; i++)
        {
            if (Neurons[^1][i].FirstSpikeT < firstSpikeTime)
            {
                firstSpikeTime = Neurons[^1][i].FirstSpikeT;
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
            error += Math.Pow(Neurons[^1][i].FirstSpikeT - desiredSpikeTimes[i], 2);
        }

        return 0.5 * error;
    }

    public void Backward(double[] desiredSpikeTimes)
    {
        // Layers
        for (int l = Dimensions.Length - 1; l > 0; l--)
        {
            // Neurons
            for (int i = 0; i < Dimensions[l]; i++)
            {
                Neurons[l][i].UpdateInputWeights(desiredSpikeTimes[i], LEARNING_RATE, DECAY_TIME, SYN_PER_NEURON);
            }
        }
    }

    public void Report(int epoch)
    {
        string reportNeurons = "";
        string reportSynapses = "";

        // Layers
        for (int l = 0; l < Dimensions.Length; l++)
        {
            // Pre-Neurons
            for (int i = 0; i < Dimensions[l]; i++)
            {
                reportNeurons += $"Neuron[L{l}N{i}] spikes (first|total): {Neurons[l][i].FirstSpikeT}|{Neurons[l][i].SpikeTs.Count}\n";
                reportNeurons += "Spike graph: ";
                for (double t = 0; t < INTERVAL_DURATION; t += TIME_STEP)
                {
                    // Fix floating point error
                    t = Math.Round(t, 1);
                    reportNeurons += Neurons[l][i].SpikeTs.Any(ts => Math.Round(ts, 1) == t) ? "|" : ".";
                }
                reportNeurons += "\n";

                if (l != Dimensions.Length - 1)
                {
                    // Post-Neurons
                    for (int j = 0; j < Dimensions[l + 1]; j++)
                    {
                        // Synapses
                        for (int s = 0; s < SYN_PER_NEURON; s++)
                        {
                            reportSynapses += $"Synapse {s} between [L{l}N{i}] and [L{l + 1}N{j}] weight: {Neurons[l][i].SynapsesOut[j * s].Weight}\n";
                        }
                    }
                }
            }

            reportNeurons += "------------------------------------------------------------\n";

            if (l != Dimensions.Length - 1)
            {
                reportSynapses += "------------------------------------------------------------\n";
            }
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

sealed class Neuron
{
    public const double TAU_R = 7;
    public const double TAU_M = 4;
    public const double TAU_S = 2;

    public double Potential { get; set; }
    public List<double> SpikeTs { get; set; }
    public double FirstSpikeT => SpikeTs.Count > 0 ? SpikeTs[0] : -1;
    public double LastSpikeT => SpikeTs.Count > 0 ? SpikeTs[^1] : -1;
    public List<Synapse> SynapsesIn { get; set; }
    public List<Synapse> SynapsesOut { get; set; }

    public Neuron()
    {
        SpikeTs = new();
        SynapsesIn = new();
        SynapsesOut = new();
    }

    public void Reset()
    {
        SpikeTs.Clear();
    }

    public void EncodeInput(double input, double maxDelay)
    {
        SpikeTs.Add(maxDelay * input / 2);
    }

    public bool CalculatePotential(double t, double threshold, double decayTime)
    {
        Potential = 0;
        foreach (double spikeT in SpikeTs)
        {
            Potential += ExponentialDecay(t - spikeT, threshold);
        }

        foreach (Synapse syn in SynapsesIn.Where(s => s.NeuronPre.SpikeTs.Count > 0))
        {
            foreach (double spikeT in syn.NeuronPre.SpikeTs)
            {
                Potential += syn.Weight * SpikeResponseFunction_24(t - spikeT - syn.Delay, decayTime);
            }
        }

        if (Potential >= threshold)
        {
            SpikeTs.Add(t);
            return true;
        }

        return false;
    }

    private static double ExponentialDecay(double s, double threshold, double tauR = TAU_R) =>
        s >= 0 ? -threshold * Math.Exp(-s / tauR) : 0;

    public static double SpikeResponseFunction_23(double s, double tauM = TAU_M, double tauS = TAU_S) =>
        s >= 0 ? (Math.Exp(-s / tauM) - Math.Exp(-s / tauS)) : 0;

    public static double SpikeResponseFunction_24(double s, double tau) =>
        s >= 0 ? s / tau * Math.Exp(1 - s / tau) : 0;

    public void UpdateInputWeights(double desiredSpikeTime, double learningRate, double decayTime, int synPerNeuron)
    {
        //foreach(Synapse preSyn in SynapsesIn)
        //{
        //    preSyn.UpdateWeight(desiredSpikeTime, learningRate, decayTime);
        //}
        //return;

        ///// delta_output_layer /////
        double s = 0;
        if (FirstSpikeT != -1)
        {
            // Pre-Neuron
            for (int neuronPreI = 0; neuronPreI < SynapsesIn.Count / synPerNeuron; neuronPreI++)
            {
                // Synapse per pre-neuron
                int neuronSynStartIndex = neuronPreI * synPerNeuron;
                if (SynapsesIn[neuronSynStartIndex].NeuronPre.FirstSpikeT != -1)
                {
                    for (int sI = 0; sI < synPerNeuron; sI++)
                    {
                        double spikeDistance = FirstSpikeT - SynapsesIn[neuronPreI * sI].NeuronPre.FirstSpikeT - SynapsesIn[neuronPreI * sI].Delay;
                        double weight = SynapsesIn[neuronPreI * sI].Weight;
                        if (spikeDistance == 0)
                        {
                            s += weight * Math.E / decayTime;
                        }
                        else if (spikeDistance > 0)
                        {
                            s += weight * (1 / spikeDistance - 1 / decayTime) * SpikeResponseFunction_24(spikeDistance, decayTime);
                        }
                    }
                }
            }
        }
        double delta = s != 0 ? (desiredSpikeTime - FirstSpikeT) / s : 0;

        ///// backward_prop /////
        double[] deltaWs = new double[SynapsesIn.Count];
        // Pre-Neuron
        for (int neuronPreI = 0; neuronPreI < SynapsesIn.Count / synPerNeuron; neuronPreI++)
        {
            // Synapse per per-neuron
            int neuronSynStartIndex = neuronPreI * synPerNeuron;
            if (SynapsesIn[neuronSynStartIndex].NeuronPre.FirstSpikeT != -1)
            {
                for (int sI = 0; sI < synPerNeuron; sI++)
                {
                    deltaWs[neuronSynStartIndex + sI] = -learningRate 
                        * SpikeResponseFunction_24(FirstSpikeT - SynapsesIn[neuronPreI * sI].NeuronPre.FirstSpikeT - SynapsesIn[neuronPreI * sI].Delay, decayTime) 
                        * delta;
                }
            }
        }

        for (int i = 0; i < SynapsesIn.Count; i++)
        {
            SynapsesIn[i].Weight += deltaWs[i];
            if (SynapsesIn[i].Weight < 0)
            {
                SynapsesIn[i].Weight = 0;
            }
        }
    }
}

sealed class Synapse
{
    public double Weight { get; set; }
    public double Delay { get; set; }
    public Neuron NeuronPre { get; set; }
    public Neuron NeuronPost { get; set; }

    public Synapse(Neuron neuronPre, Neuron neuronPost, double weight, double delay)
    {
        NeuronPre = neuronPre;
        NeuronPost = neuronPost;
        Weight = weight;
        Delay = delay;
    }

    [Obsolete]
    public void UpdateWeight(double desiredSpikeTimePostNeuron, double learningRate, double decayTime)
    {
        double dividend = 0;
        foreach (double spikeT in NeuronPre.SpikeTs)
        {
            dividend += Neuron.SpikeResponseFunction_24(NeuronPost.FirstSpikeT - spikeT - Delay, decayTime);
        }
        dividend *= -1;

        double divisor = 1;
        foreach (double spikeT in NeuronPre.SpikeTs)
        {
            double spikeDistance = NeuronPost.FirstSpikeT - spikeT - Delay;

            if (spikeDistance == 0)
            {
                divisor += Weight * Math.E / decayTime;
            }
            else if (spikeDistance > 0)
            {
                divisor += Weight * (1 / spikeDistance - 1 / decayTime) * Neuron.SpikeResponseFunction_24(spikeDistance, decayTime);
            }
        }

        double factor = -learningRate * (NeuronPost.FirstSpikeT - desiredSpikeTimePostNeuron);
        double deltaW = dividend / divisor * factor;

        if (!double.IsNaN(deltaW))
        {
            Weight += deltaW;
        }

        if (Weight < 0)
        {
            Weight = 0;
        }
    }
}