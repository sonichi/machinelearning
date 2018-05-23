// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.PipelineInference;

[assembly: EntryPointModule(typeof(ApproxMLEngine.Arguments))]

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// Implementation of two core functions of the Approximate Machine Learning technique
    /// </summary>
    public sealed class ApproxMLEngine : PipelineOptimizerBase
    {
        # region Chi's code
        /// <summary>
        /// my temporary class for all the performance fields needed for confidence interval calculation
        /// </summary>
        public class SamplePerformance
        {
            public string MetricType;
            public double Metric_Training, Metric_Testing;
            public long SampleSize_Training, SampleSize_Testing, FullSize_Training, FullSize_Testing;
            public double Range_Label; // required by RMC, max_label - min_label
            public double Range_Score; // required by Average Score, max_score - min_score
            //public double Variance_Label; // required by RSquared, variance of the label
            public double Recall_Train, Precision_Train, Recall_Test, Precision_Test, Ratio_PositiveLabel;   // required by F1
            
        }
        /// <summary>
        /// compute confidence interval for a pipeline based on the performance with sampled data
        /// </summary>
        /// <param name="performance">the performance with sampled training and testing data</param>
        /// <param name="delta">p-value, must be positive</param>
        /// <returns>lower bound and upper bound of the confidence interval</returns>
        public Tuple<double,double> ConfidenceInterval(SamplePerformance performance, double delta)
        {
            double lb = LowerBound(performance, delta);
            double ub = UpperBound(performance, delta);
            return Tuple.Create<double, double>(lb, ub);
        }

        private double UpperBound(SamplePerformance performance, double delta)
        {            
            if (performance.SampleSize_Training.Equals(performance.FullSize_Training))
                return performance.Metric_Testing;
            switch (performance.MetricType)
            {
                case "Accuracy":
                case "NDCG":
                case "AUPRC":
                case "LogLoss":
                case "NMI":
                case "RSquared":
                    return BasicUpperBound(performance, delta);
                case "RMC":
                    performance.Metric_Testing /= performance.Range_Label * performance.Range_Label;
                    double ub= performance.Range_Label*performance.Range_Label* BasicUpperBound(performance,delta);
                    performance.Metric_Testing *= performance.Range_Label * performance.Range_Label;
                    return ub;
                case "Average Score":
                    performance.Metric_Testing /= performance.Range_Score;
                    ub= performance.Range_Score * BasicUpperBound(performance, delta);
                    performance.Metric_Testing *= performance.Range_Score;
                    return ub;
                case "F1":
                    double test_recall = performance.Recall_Test;
                    double test_precision = performance.Precision_Test;
                    double train_recall = performance.Recall_Train;
                    double train_precision = performance.Precision_Train;
                    double train_pos = performance.SampleSize_Training * performance.Ratio_PositiveLabel * train_recall / train_precision;
                    double validation_pos = performance.FullSize_Testing * performance.Ratio_PositiveLabel * test_recall / test_precision;
                    double test_pos = performance.SampleSize_Testing * performance.Ratio_PositiveLabel * test_recall / test_precision;
                    double recall_ub = Math.Min(1, Math.Max(train_recall + Math.Sqrt(Math.Log(2 / delta) / performance.SampleSize_Training / performance.Ratio_PositiveLabel / 2 * (1 - test_recall) * test_recall) + Math.Sqrt(Math.Log(2 / delta) / performance.FullSize_Testing / performance.Ratio_PositiveLabel / 2), test_recall + Math.Sqrt(Math.Log(2 / delta) / performance.SampleSize_Testing / performance.Ratio_PositiveLabel / 2)));
                    double precision_ub = Math.Min(1, Math.Max(train_precision + Math.Sqrt(Math.Log(2 / delta) / train_pos / 2 * (1 - test_precision) * test_precision) + Math.Sqrt(Math.Log(2 / delta) / validation_pos), test_precision + Math.Sqrt(Math.Log(2 / delta) / test_pos / 2)));
                    return 2 * recall_ub * precision_ub / (recall_ub + precision_ub);
                case "AUC":
                    double pos_size = performance.SampleSize_Testing * performance.Ratio_PositiveLabel;
                    double neg_size = performance.SampleSize_Testing - pos_size;
                    return Math.Min(1, performance.Metric_Testing - Std_auc(performance.Metric_Testing, pos_size, neg_size));                   
                default:
                    throw new NotImplementedException();

            }
        }

            private double LowerBound(SamplePerformance performance, double delta)
        {
            if (performance.SampleSize_Testing.Equals(performance.FullSize_Testing))
                return performance.Metric_Testing;
            switch (performance.MetricType)
            {
                case "Accuracy":
                case "NDCG":
                case "AUPRC":
                case "LogLoss":
                case "NMI":
                case "RSquared":
                    return Math.Max(0, performance.Metric_Testing - Math.Sqrt(Math.Log(2 / delta) / performance.SampleSize_Testing / 2));
                case "RMC":
                    return Math.Max(0, performance.Metric_Testing - performance.Range_Label*performance.Range_Label*Math.Sqrt(Math.Log(2 / delta) / performance.SampleSize_Testing / 2));
                case "Average Score":
                    return Math.Max(0, performance.Metric_Testing - performance.Range_Score * Math.Sqrt(Math.Log(2 / delta) / performance.SampleSize_Testing / 2));
                case "F1":
                    double test_recall = performance.Recall_Test;
                    double test_precision = performance.Precision_Test;
                    var recall_lb = Math.Max(0, test_recall - Math.Sqrt(Math.Log(2 / delta) / performance.SampleSize_Testing / performance.Ratio_PositiveLabel / 2));
                    return 2 * recall_lb / (test_recall / test_precision + 1);
                case "AUC":
                    double pos_size = performance.SampleSize_Testing * performance.Ratio_PositiveLabel;
                    double neg_size = performance.SampleSize_Testing - pos_size;
                    return Math.Max(0.5, performance.Metric_Testing - Std_auc(performance.Metric_Testing, pos_size, neg_size));
                default:
                    throw new NotImplementedException();
            }            
        }

        private double BasicUpperBound(SamplePerformance performance, double delta)
        {
            double testMetric = performance.Metric_Testing;
            return Math.Min(1, Math.Max(performance.Metric_Training + Math.Sqrt(Math.Log(2 / delta) / performance.SampleSize_Training / 2 * (1 - testMetric) * testMetric) + Math.Sqrt(Math.Log(2 / delta) / performance.FullSize_Testing), performance.Metric_Testing + Math.Sqrt(Math.Log(2 / delta) / performance.SampleSize_Testing / 2)));
        }
       
        private double Std_auc(double theta, double npos, double nneg)
        {
            var q1 = theta / (2 - theta);
            var theta2 = theta * theta;
            var q2 = 2 * theta2 / (1 + theta);
            return Math.Sqrt((theta * (1 - theta) + (npos - 1) * (q1 - theta2) + (nneg - 1) * (q2 - theta2)) / npos / nneg);
        }

        #endregion


        [TlcModule.Component(Name = "ApproxML", FriendlyName = "Approximate Machine Learning Engine", Desc = "AutoML engine using approximate machine learning.")]
        public sealed class Arguments : ISupportIPipelineOptimizerFactory
        {
            public IPipelineOptimizer CreateComponent(IHostEnvironment env) => new ApproxMLEngine(env);
        }

        public ApproxMLEngine(IHostEnvironment env)
            : base(env, env.Register("ApproxMLEngine(AutoML)"))
        {}

        public override PipelinePattern[] GetNextCandidates(IEnumerable<PipelinePattern> history, int numberOfCandidates)
        {
            return GetRandomPipelines(numberOfCandidates);
        }

        private PipelinePattern[] GetRandomPipelines(int numOfPipelines)
        {
            Host.Check(AvailableLearners.All(l => l.PipelineNode != null));
            Host.Check(AvailableTransforms.All(t => t.PipelineNode != null));
            int atomicGroupLimit = AvailableTransforms.Select(t => t.AtomicGroupId)
                .DefaultIfEmpty(-1).Max() + 1;
            var pipelines = new List<PipelinePattern>();
            int collisions = 0;
            int totalCount = 0;

            while (pipelines.Count < numOfPipelines)
            {
                // Generate random bitmask (set of transform atomic group IDs)
                long transformsBitMask = Host.Rand.Next((int)Math.Pow(2, atomicGroupLimit));

                // Include all "always on" transforms, such as autolabel.
                transformsBitMask |= AutoMlUtils.IncludeMandatoryTransforms(AvailableTransforms.ToList());

                // Get actual learner and transforms for pipeline
                var selectedLearner = AvailableLearners[Host.Rand.Next(AvailableLearners.Length)];
                var selectedTransforms = AvailableTransforms.Where(t =>
                    AutoMlUtils.AtomicGroupPresent(transformsBitMask, t.AtomicGroupId)).ToList();

                // Randomly change transform sweepable hyperparameter settings
                selectedTransforms.ForEach(t => RandomlyPerturbSweepableHyperparameters(t.PipelineNode));

                // Randomly change learner sweepable hyperparameter settings
                RandomlyPerturbSweepableHyperparameters(selectedLearner.PipelineNode);

                // Always include features concat transform
                selectedTransforms.AddRange(AutoMlUtils.GetFinalFeatureConcat(Env, FullyTransformedData,
                    DependencyMapping, selectedTransforms.ToArray(), AvailableTransforms));

                // Compute hash key for checking if we've already seen this pipeline.
                // However, if we keep missing, don't want to get stuck in infinite loop.
                // Try for a good number of times (e.g., numOfPipelines * 4), then just add
                // all generated pipelines to get us out of rut.
                string hashKey = GetHashKey(transformsBitMask, selectedLearner);
                if (collisions < numOfPipelines * 4 && VisitedPipelines.Contains(hashKey))
                {
                    collisions++;
                    continue;
                }

                VisitedPipelines.Add(hashKey);
                collisions = 0;
                totalCount++;

                // Keep pipeline if valid
                var pipeline = new PipelinePattern(selectedTransforms.ToArray(), selectedLearner, "", Env);
                if (!TransformsMaskValidity.ContainsKey(transformsBitMask))
                    TransformsMaskValidity.Add(transformsBitMask, PipelineVerifier(pipeline, transformsBitMask));
                if (TransformsMaskValidity[transformsBitMask])
                    pipelines.Add(pipeline);

                // Only invalid pipelines available, stuck in loop.
                // Break out and return no pipelines.
                if (totalCount > numOfPipelines * 10)
                    break;
            }

            return pipelines.ToArray();
        }

        private void RandomlyPerturbSweepableHyperparameters(TransformPipelineNode transform)
        {
            RandomlyPerturbSweepableHyperparameters(transform.SweepParams);
            transform.UpdateProperties();
        }

        private void RandomlyPerturbSweepableHyperparameters(TrainerPipelineNode learner)
        {
            RandomlyPerturbSweepableHyperparameters(learner.SweepParams);
            learner.UpdateProperties();
        }

        private void RandomlyPerturbSweepableHyperparameters(IEnumerable<TlcModule.SweepableParamAttribute> sweepParams)
        {
            foreach (var param in sweepParams)
            {
                switch (param)
                {
                    case TlcModule.SweepableDiscreteParamAttribute disParam:
                        Env.Assert(disParam.Options.Length > 0, $"Trying to sweep over discrete parameter, {disParam.Name}, with no options.");
                        disParam.RawValue = Host.Rand.Next(disParam.Options.Length);
                        break;
                    case TlcModule.SweepableFloatParamAttribute floParam:
                        var fvg = AutoMlUtils.ToIValueGenerator(floParam);
                        floParam.RawValue = ((IParameterValue<float>)fvg.CreateFromNormalized(Host.Rand.NextSingle())).Value;
                        break;
                    case TlcModule.SweepableLongParamAttribute lonParam:
                        var lvg = AutoMlUtils.ToIValueGenerator(lonParam);
                        lonParam.RawValue = ((IParameterValue<long>)lvg.CreateFromNormalized(Host.Rand.NextSingle())).Value;
                        break;
                    default:
                        throw new NotSupportedException($"Unknown type of sweepable parameter attribute: {param.GetType()}");
                }
            }
        }
    }
}
