using SnnTest;

namespace Snn2
{
    internal static class TestPoisson
    {
        public static void Test()
        {
            const int SEED = 2;
            const int EPOCHS = 50;
            const int TEST_COUNT = 100;
            Random RND = new(SEED);

            Network network = new(RND, 1);
            Console.WriteLine($"Fitting Poisson for {EPOCHS} epochs");

            double lowestError = double.MaxValue;
            double avgError = 0;
            (List<double>[] samples, bool[] labels) = DataManager.GenerateSinglePoissonBenchmarkData(RND);
            for (int epoch = 0; epoch < EPOCHS; epoch++)
            {
                double[] desiredSpikeTimes; ;
                desiredSpikeTimes = labels[1] ? new double[] { 22, 17, 22, 22 } : new double[] { 22, 22, 22, 22 };

                double error = 0;
                network.Reset();
                network.Forward(samples);
                error += network.CalculateError(desiredSpikeTimes);
                network.Backward(desiredSpikeTimes);

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
                (List<double>[] samplesTest, bool[] labelsTest) = DataManager.GenerateSinglePoissonBenchmarkData(RND);

                network.Reset();
                network.Forward(samplesTest);

                bool prediction = network.GetCurrentlyPredictedClass() == 1;
                if (prediction)
                {
                    if (labelsTest[1])
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
                    if (labelsTest[1])
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