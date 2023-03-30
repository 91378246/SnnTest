namespace Snn2;

public sealed class Neuron
{
    public List<Synapse> SynapsesPre { get; set; } = new();
    public List<Synapse> SynapsesPost { get; set; } = new();
    public double Potential { get; set; }
    public List<double> Spikes { get; set; } = new();

    private List<Neuron> NeuronsPre => SynapsesPre.Where(s => s.NeuronPost == this).Select(s => s.NeuronPre).Distinct().ToList();
    private List<Neuron> NeuronsPost => SynapsesPost.Where(s => s.NeuronPre == this).Select(s => s.NeuronPost).Distinct().ToList();


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
    /// Spike response function ε(s) (4)
    /// </summary>
    /// <param name="s"></param>
    /// <returns></returns>
    public static double SpikeResponseFunction(double s) =>
        s >= 0 ? (Math.Exp(-s / Parameters.TAU_M) - Math.Exp(-s / Parameters.TAU_S)) : 0;

    /// <summary>
    /// Spike response function ε(s) (4) derived with respect to s
    /// </summary>
    /// <param name="s"></param>
    /// <returns></returns>
    public static double SpikeResponseFunctionDerived(double s) =>
        s >= 0 ? -Math.Exp(-s / Parameters.TAU_M) / Parameters.TAU_M + Math.Exp(-s / Parameters.TAU_S) / Parameters.TAU_S : 0;

    /// <summary>
    /// Exponential decay η(s) (5)
    /// </summary>
    /// <param name="s"></param>
    /// <returns></returns>
    public static double ExponentialDecay(double s) =>
        s >= 0 ? -Parameters.LEARNING_RATE * Math.Exp(-s / Parameters.TAU_R) : 0;

    /// <summary>
    /// Exponential decay η(s) (5) derived with respect to s
    /// </summary>
    /// <param name="s"></param>
    /// <returns></returns>
    public static double ExponentialDecayDerived(double s) =>
        s >= 0 ? -Parameters.LEARNING_RATE * Math.Exp(-s / Parameters.TAU_R) / Parameters.TAU_R : 0;

    /// <summary>
    /// Potential (3)
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
        foreach (Neuron preNeuron in NeuronsPre)
        {
            foreach (double spikePre in preNeuron.Spikes)
            {
                foreach (Synapse synapsePre in preSynapses.Where(s => s.NeuronPre == preNeuron))
                {
                    secondSum += synapsePre.Weight * SpikeResponseFunction(t - spikePre - synapsePre.Delay);
                }
            }
        }

        Potential = firstSum + secondSum; // Math.Max(firstSum + secondSum, 0);

        if (Potential > Parameters.THRESHOLD)
        {
            Spikes.Add(t);
            return true;
        }

        return false;
    }

    /// <summary>
    /// ∆w^k_ih (9)
    /// </summary>
    /// <param name="desiredFirstSpikeTime"></param>
    public void UpdateInputWeights(double? desiredFirstSpikeTime = null)
    {
        if (Spikes.Count == 0)
        {
            return;
        }

        Dictionary<Synapse, double> gradients = new();
        Dictionary<Synapse, double> deltaWs = new();
        foreach (Neuron neuronPre in NeuronsPre)
        {
            foreach (Synapse synPre in SynapsesPre.Where(s => s.NeuronPre == neuronPre))
            {
                foreach (double spikeThis in Spikes)
                {
                    deltaWs[synPre] -= Parameters.LEARNING_RATE * CalculateDeDt(spikeThis, desiredFirstSpikeTime) * CalculateDtDw(synPre, spikeThis);
                }
            }
        }

    }

    // (10)
    private double CalculateDtDw(Synapse synPre, double spike)
    {
        return -CalculateDuDw(synPre, spike) / CalculateDuDt(spike);
    }

    // (11)
    private double CalculateDuDw(Synapse synPre, double spike)
    {
        double result = 0;
        foreach (double spikePre in synPre.NeuronPre.Spikes)
        {
            result += SpikeResponseFunction(spike - spikePre - synPre.Delay);
        }

        foreach (double spikeThis in Spikes)
        {
            if (spikeThis < spike)
            {
                result += -ExponentialDecayDerived(spike - spikeThis) * CalculateDtDw(synPre, spikeThis);
            }
        }

        return result;
    }

    // (12)
    public double CalculateDuDt(double spike)
    {
        double result = 0;
        foreach (Synapse synPre in SynapsesPre)
        {
            foreach (double spikePre in synPre.NeuronPre.Spikes)
            {
                result += synPre.Weight * SpikeResponseFunctionDerived(spike - spikePre - synPre.Delay);
            }
        }

        foreach (double spikeThis in Spikes)
        {
            if (spikeThis < spike)
            {
                result += ExponentialDecayDerived(spike - spikeThis);
            }
        }

        if (result < 0.1)
        {
            result = 0.1;
        }

        return result;
    }

    // (13)
    public double CalculateDeDt(double spike, double? desiredFirstSpikeTime)
    {
        if (desiredFirstSpikeTime != null && spike == Spikes.First())
        {
            return spike = desiredFirstSpikeTime.Value;
        }

        double result = 0;
        foreach (Neuron neuronPost in NeuronsPost)
        {
            foreach (double spikePost in neuronPost.Spikes)
            {
                if (spikePost > spike)
                {
                    result += neuronPost.CalculateDeDt(spikePost, desiredFirstSpikeTime) * CalculateDPostTdt(neuronPost, spike, spikePost);
                }
            }
        }

        return result;
    }

    // (14)
    private double CalculateDPostTdt(Neuron neuronPost, double spike, double spikePost)
    {
        return -CalculateDPostUdt(neuronPost, spike, spikePost) / neuronPost.CalculateDuDt(spike);
    }

    // (15)
    double CalculateDPostUdt(Neuron neuronPost, double spike, double spikePost)
    {
        double result = 0;
        foreach (Synapse synPost in neuronPost.SynapsesPre.Where(s => s.NeuronPre == this))
        {
            result -= synPost.Weight * SpikeResponseFunctionDerived(spikePost - spike - synPost.Delay);
        }

        foreach (double spikeFromPostNeuron in neuronPost.Spikes)
        {
            if (spikeFromPostNeuron < spikePost)
            {
                result -= ExponentialDecayDerived(spikePost - spikeFromPostNeuron) * CalculateDPostTdt(neuronPost, spike, spikeFromPostNeuron);
            }
        }

        return result;
    }

    public void Reset()
    {
        Potential = 0;
        Spikes = new();
    }
}