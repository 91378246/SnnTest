namespace Snn2.Test
{
    public sealed class DataManagerTest
    {
        [Test]
        public void GenerateXorDataTest()
        {
            Random rnd = new();
            for (int i = 0; i < 100; i++)
            {
                (List<double>[] inputs, bool[] label) = DataManager.GenerateXorData(count: 2, tMax: 100, rnd: rnd);
                for (int r = 0; r < 2; r++)
                {
                    if (inputs[0][r] == inputs[1][r])
                    {
                        Assert.That(label[r], Is.False);
                    }
                    else
                    {
                        Assert.That(label[r], Is.True);
                    }
                }
            }
        }

        [Test]
        public void GenerateSinglePoissonBenchmarkDataTest()
        {
            Random rnd = new();
            (List<double>[] inputs, _) = DataManager.GenerateSinglePoissonBenchmarkData(rnd: rnd);

            int avg = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                avg += inputs[i].Count;
            }

            avg /= inputs.Length;
            Assert.LessOrEqual(3, avg);
            Assert.GreaterOrEqual(3.5, avg);
        }
    }
}