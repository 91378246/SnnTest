using SnnTest;

namespace Snn2
{
    internal static class TestXor
    {
        public static void Test()
        {
            const int SEED = 2;
            const int EPOCHS = 100;
            const int SAMPLE_COUNT = 1000;
            const int TEST_COUNT = 100;
            Random RND = new(SEED);

            Network network = new(RND, 1);
            Console.WriteLine($"Fitting XOR for {EPOCHS} epochs");

            double lowestError = double.MaxValue;
            double avgError = 0;
            for (int epoch = 0; epoch < EPOCHS; epoch++)
            {
                (List<double>[] samples, bool[] labels) = DataManager.GenerateXorData(SAMPLE_COUNT, tMax: Parameters.INTERVAL_DURATION, RND);
                List<double[]> desiredSpikeTimes = new();
                for (int i = 0; i < labels.Length; i++)
                {
                    desiredSpikeTimes.Add(labels[i] ? new double[] { 22, 17 } : new double[] { 17, 22 });
                }

                double error = 0;
                for (int i = 0; i < SAMPLE_COUNT; i++)
                {
                    network.Reset();
                    network.Forward(samples);
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
            for (int i = 0; i < TEST_COUNT; i++)
            {
                (List<double>[] samplesTest, bool[] labelsTest) = DataManager.GenerateXorData(TEST_COUNT, tMax: Parameters.INTERVAL_DURATION, RND);
                network.Reset();
                network.Forward(samplesTest);

                bool prediction = Convert.ToBoolean(network.GetCurrentlyPredictedClass());
                if (prediction)
                {
                    if (labelsTest[i])
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
                    if (labelsTest[i])
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

        }
    }
}