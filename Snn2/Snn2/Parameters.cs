namespace Snn2
{
    public static class Parameters
    {
        // i: Pre-Neuron
        // j: Post-Neuron
        // Γ_j: All presynaptic neurons of j
        // u_j: Current potential of neuron j
        // t^(g): Spike-time pre neuron
        // t^(f): Spike-time post neuron
        // t^(1): First spike-time
        // t^^(1): Desired first spike-time
        // n: Number of spikes
        // F_i: Spike-train (chronologically ordered)
        // w_ji: Weight
        // d_ji: Delay
        // ε: Spike response function, describes the effect the presynaptic spike has on the potential of the postsynaptic neuron
        // s: Spike-time post neuron - Spike-time pre neuron - delay
        // H(s): Heavy-side step function (H(s) = 0 for s ≤ 0 and H(s) = 1 for s > 0)
        // l: Set of delays
        // d: Delay
        // k: Synapse

        /// <summary>
        /// ϑ
        /// </summary>
        public const double THRESHOLD = 1;
        /// <summary>
        /// η
        /// </summary>
        public const double LEARNING_RATE = 0.001;
        /// <summary>
        /// τ_m: Time-constant with 0 < τ_s < τ_m
        /// </summary>
        public const double TAU_M = 4;
        /// <summary>
        /// τ_s: Time-constant with 0 < τ_s < τ_m
        /// </summary>
        public const double TAU_S = 2;
        /// <summary>
        /// τ_r: Time-constant
        /// </summary>
        public const double TAU_R = 20;

        public const int SYN_PER_NEURON = 3;
        public const int INTERVAL_DURATION = 25;
        public const double MIN_DELAY = 1;
        public const double MAX_DELAY = 8;
        public const double DECAY_TIME = 7;
        public const double TIME_STEP = 0.1;
    }
}
