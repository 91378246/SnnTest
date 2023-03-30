namespace Snn2;

public sealed class Neuron
{
    public double Potential { get; set; }
    public List<double> Spikes { get; set; } = new();

    /// <summary>
    /// Calculates spikeTimePostNeuron - spikeTimePreNeuron - delay
    /// </summary>
    /// <param name="spikeTimePreNeuron"></param>
    /// <param name="spikeTimePostNeuron"></param>
    /// <param name="delay"></param>
    /// <returns></returns>
    public static double CalculateS(double spikeTimePreNeuron, double spikeTimePostNeuron, double delay) =>
         spikeTimePostNeuron - spikeTimePreNeuron - delay;

    /// <summary>
    /// Spike response function ε(s) (2.3)
    /// </summary>
    /// <param name="s"></param>
    /// <returns></returns>
    public static double SpikeResponseFunction(double s) =>
        s >= 0 ? (Math.Exp(-s / Parameters.TAU_M) - Math.Exp(-s / Parameters.TAU_S)) : 0;

    /// <summary>
    /// Spike response function ε(s) (2.3) derived in respect to s
    /// </summary>
    /// <param name="s"></param>
    /// <returns></returns>
    public static double SpikeResponseFunctionDerived(double s) =>
        s >= 0 ? (-1 / Parameters.TAU_M) * Math.Exp(-s / Parameters.TAU_M) + (1 / Parameters.TAU_S) * Math.Exp(-s / Parameters.TAU_S) : 0;

    /// <summary>
    /// Exponential decay η(s) (2.5)
    /// </summary>
    /// <param name="s"></param>
    /// <returns></returns>
    public static double ExponentialDecay(double s) =>
        s >= 0 ? -Parameters.THRESHOLD * Math.Exp(-s / Parameters.TAU_R) : 0;

    /// <summary>
    /// Potential (2.7)
    /// </summary>
    /// <param name="t"></param>
    /// <param name="preSynapses"></param>
    public bool UpdatePotential(double t, List<Synapse> preSynapses)
    {
        double firstSum = 0;
        foreach (double spike in Spikes)
        {
            firstSum += ExponentialDecay(t - spike);
        }

        double secondSum = 0;
        List<Neuron> preNeurons = preSynapses.Where(s => s.NeuronPost == this).Select(s => s.NeuronPre).Distinct().ToList();
        if (preNeurons.Count == 0)
        {
            throw new Exception("No pre-neurons detected");
        }
        for (int preNeuronI = 0; preNeuronI < preNeurons.Count; preNeuronI++)
        {
            foreach (double spikePre in preNeurons[preNeuronI].Spikes)
            {
                foreach (Synapse synapsePre in preSynapses.Where(s => s.NeuronPre == preNeurons[preNeuronI]))
                {
                    secondSum += synapsePre.Weight * SpikeResponseFunction(t - spikePre - synapsePre.Delay);
                }
            }
        }

        Potential = firstSum + secondSum; // Math.Max(firstSum + secondSum, 0);

        if (Potential >= Parameters.THRESHOLD)
        {
            Spikes.Add(t);
            return true;
        }

        return false;
    }

    /// <summary>
    /// ∆w^k_ji (3.14)
    /// </summary>
    /// <param name="preSynapses"></param>
    public void UpdateInputWeights(List<Synapse> preSynapses, double desiredFirstSpikeTime)
    {
        if (Spikes.Count == 0)
        {
            return;
        }

        List<Neuron> preNeurons = preSynapses.Where(s => s.NeuronPost == this).Select(s => s.NeuronPre).Distinct().ToList();
        for (int preNeuronI = 0; preNeuronI < preNeurons.Count; preNeuronI++)
        {
            double gradient = CalculateGradient(preSynapses.Where(s => s.NeuronPre == preNeurons[preNeuronI] && s.NeuronPost == this).ToList());
            double deltaW = -Parameters.LEARNING_RATE * gradient * (Spikes.FirstOrDefault() - desiredFirstSpikeTime);
            foreach (Synapse synapsePre in preSynapses.Where(s => s.NeuronPre == preNeurons[preNeuronI]))
            {               
                synapsePre.Weight += deltaW;
            }
        }
    }

    /// <summary>
    /// (3.13)
    /// </summary>
    /// <param name="preSynapsesFromSinglePreNeuron"></param>
    /// <returns></returns>
    private double CalculateGradient(List<Synapse> preSynapsesFromSinglePreNeuron)
    {
        double numerator = 0;
        double denominator = 0;
        foreach (Synapse synapsePre in preSynapsesFromSinglePreNeuron)
        {
            foreach (double spikePre in synapsePre.NeuronPre.Spikes)
            {
                double s = Spikes.FirstOrDefault() - spikePre - synapsePre.Delay;
                numerator += SpikeResponseFunction(s);
                denominator += synapsePre.Weight * SpikeResponseFunctionDerived(s);
            }
        }

        double gradient = -numerator / denominator;
        if (gradient < 0.1 || double.IsNaN(gradient))
        {
            gradient = 0.1;
        }

        return gradient;
    }

    public void Reset()
    {
        Potential = 0;
        Spikes = new();
    }
}