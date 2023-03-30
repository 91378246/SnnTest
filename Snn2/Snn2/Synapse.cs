namespace Snn2;

public sealed class Synapse
{
    public double Weight { get; set; }
    public double Delay { get; set; }
    public Neuron NeuronPre { get; set; } = null!;
    public Neuron NeuronPost { get; set; } = null!;
}
