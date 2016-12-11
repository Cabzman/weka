/**
 * Created by Cabz on 11/28/2016.
 */

import weka.core.converters.ConverterUtils.DataSource;
import  weka.core.Instances;
import weka.classifiers.lazy.IBk;
import weka.classifiers.Evaluation;
import weka.gui.beans.Classifier;

public class Main {



    public static void main(String[] args) throws Exception {



            DataSource source = new DataSource("D:\\Libraries\\Desktop\\Bigd Project\\arff files\\abaloneTrainingSet.arff");
            DataSource source2 = new DataSource("D:\\Libraries\\Desktop\\Bigd Project\\arff files\\abaloneTestSet.arff");
            Instances dataTrain = source.getDataSet();
            Instances dataTest = source2.getDataSet();
            dataTrain.setClassIndex(1);
            dataTest.setClassIndex(1);


        weka.classifiers.Classifier cls = new IBk(3);

        cls.buildClassifier(dataTrain);
        Evaluation eval = new Evaluation(dataTrain);
        eval.evaluateModel(cls,dataTest);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));


    }
}
