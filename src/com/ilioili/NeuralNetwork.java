package com.ilioili;

import java.text.DecimalFormat;
import java.util.ArrayList;

/**
 * Created by ilioili on 2018/1/3.
 */
public class NeuralNetwork {
    public static final int WEIGHT_MAX = 10;
    private final int[] layerConfig;
    private int layerNum;
    /**
     * list_σ.get(layer)[nodeIndex]
     * 表示第(layer)层的第(nodeIndex)个节点的偏置 b, 第一层全部为0
     */
    private ArrayList<double[]> list_b;
    /**
     * list_σ.get(layer)[nodeIndex]
     * 表示第(layer)层的第(nodeIndex)个节点的输入值 ∑(wx+b)
     */
    private ArrayList<double[]> list_z;
    /**
     * list_σ.get(layer)[nodeIndex]
     * 表示第(layer)层的第(nodeIndex)个节点的激活值σ(∑(wx+b))
     */
    private ArrayList<double[]> list_σ;
    /**
     * list_δ.get(layer).get(nodeIndex)
     * 表示第(layer)层的第(nodeIndex)个节点的损失率δ
     */
    private ArrayList<double[]> list_δ;
    /**
     * list_w.get(layer).get(nodeIndex)[nodeIndexAfter]
     * 表示第(layer)层的第(nodeIndex)个节点到第(layer+1)层的第(nodeIndexAfter)个节点之间的权重w
     */
    private ArrayList<ArrayList<double[]>> list_w;
    private double cost;
    private double leanrgingRate;

    public NeuralNetwork(double learningRate, int... layerConfig) {
        this.layerConfig = layerConfig;
        this.leanrgingRate = learningRate;
        layerNum = layerConfig.length;
        list_b = new ArrayList<>(layerNum);
        list_z = new ArrayList<>(layerNum);
        list_σ = new ArrayList<>(layerNum);
        list_δ = new ArrayList<>(layerNum);//忽略输入层&和隐藏层之间的损失率
        list_w = new ArrayList<>(layerNum);

        for (int layerIndex = 0; layerIndex < layerNum; layerIndex++) {
            if (layerIndex == 0) {
                list_b.add(null);
                list_δ.add(null);
                list_z.add(null);
                list_w.add(null);
            } else {
                double[] nodeBias = new double[layerConfig[layerIndex]];
                MathHelper.random(nodeBias);
                list_b.add(nodeBias);
                list_δ.add(new double[layerConfig[layerIndex]]);
                list_z.add(new double[layerConfig[layerIndex]]);
                ArrayList<double[]> list = new ArrayList(layerConfig[layerIndex]);
                for (int nodeIndex = 0; nodeIndex < layerConfig[layerIndex]; nodeIndex++) {
                    double[] nodeWeights = new double[layerConfig[layerIndex - 1]];
                    MathHelper.random(nodeWeights);
                    list.add(nodeWeights);
                }
                list_w.add(list);
            }
            list_σ.add(new double[layerConfig[layerIndex]]);//对于输入层，input作为输出
        }

    }

    public void feedForward(double[] inputData) {
        System.arraycopy(inputData, 0, list_σ.get(0), 0, inputData.length);
        for (int layer = 1; layer < layerNum; layer++) {//从第layer+1层开始到最后一层
            int preLayer = layer - 1;
            for (int nodeIndex = 0; nodeIndex < layerConfig[layer]; nodeIndex++) {//遍历每层到节点，计算节点输入及输出
                double[] nodeInputWeitghts = list_w.get(layer).get(nodeIndex); //连接到该节点前面的所有输入权重
                double nodeBias = list_b.get(layer)[nodeIndex];
                double[] nodeSum = list_z.get(layer);
                nodeSum[nodeIndex] = 0;
                for (int i = 0; i < layerConfig[preLayer]; i++) {
                    nodeSum[nodeIndex] += nodeInputWeitghts[i] * list_σ.get(preLayer)[i];
                }
                nodeSum[nodeIndex] += nodeBias;
                list_σ.get(layer)[nodeIndex] = MathHelper.sigmod(nodeSum[nodeIndex]);
            }
        }
    }

    public void backPropergation(double[] outputLabels) {
        int lastLayerIndex = layerNum - 1;
        int lastSecondLayer = lastLayerIndex - 1;
        computeCost(outputLabels);


        //输出层
        for (int lastLayerNodeIndex = 0; lastLayerNodeIndex < layerConfig[lastLayerIndex]; lastLayerNodeIndex++) {//遍历输出节点
            double output = list_σ.get(lastLayerIndex)[lastLayerNodeIndex];
//            double δ = output * (1 - output) * (output - outputLabels[lastLayerNodeIndex]);
            double δ = (output - outputLabels[lastLayerNodeIndex]);
            list_δ.get(lastLayerIndex)[lastLayerNodeIndex] = δ;
            for (int lastSecondLayerNodeIndex = 0; lastSecondLayerNodeIndex < layerConfig[lastSecondLayer]; lastSecondLayerNodeIndex++) {
                double lastSecondLayerNodeOutput = list_σ.get(lastSecondLayer)[lastSecondLayerNodeIndex];
                list_w.get(lastLayerIndex).get(lastLayerNodeIndex)[lastSecondLayerNodeIndex] -= leanrgingRate * δ * lastSecondLayerNodeOutput;
                list_b.get(lastLayerIndex)[lastLayerNodeIndex] -= δ * leanrgingRate;
//                if (list_w.get(lastLayerIndex).get(lastLayerNodeIndex)[lastSecondLayerNodeIndex] > WEIGHT_MAX)
//                    list_w.get(lastLayerIndex).get(lastLayerNodeIndex)[lastSecondLayerNodeIndex] = WEIGHT_MAX;
//                if (list_w.get(lastLayerIndex).get(lastLayerNodeIndex)[lastSecondLayerNodeIndex] < -WEIGHT_MAX)
//                    list_w.get(lastLayerIndex).get(lastLayerNodeIndex)[lastSecondLayerNodeIndex] = -WEIGHT_MAX;
                if (list_b.get(lastLayerIndex)[lastLayerNodeIndex] > layerConfig[lastSecondLayer])
                    list_b.get(lastLayerIndex)[lastLayerNodeIndex] = layerConfig[lastSecondLayer];
                else if (list_b.get(lastLayerIndex)[lastLayerNodeIndex] < -layerConfig[lastSecondLayer])
                    list_b.get(lastLayerIndex)[lastLayerNodeIndex] = -layerConfig[lastSecondLayer];

            }
        }
        //隐藏层
        for (int layerIndex = lastSecondLayer; layerIndex > 0; layerIndex--) {
            int nextLayerIndex = layerIndex + 1;
            int preLayerIndex = layerIndex - 1;
            for (int nodeIndex = 0; nodeIndex < layerConfig[layerIndex]; nodeIndex++) {
                double δ = 0;
                for (int nextLayerNodeIndex = 0; nextLayerNodeIndex < layerConfig[nextLayerIndex]; nextLayerNodeIndex++) {
                    δ += list_δ.get(nextLayerIndex)[nextLayerNodeIndex] * list_w.get(nextLayerIndex).get(nextLayerNodeIndex)[nodeIndex];
                }
                double output = list_σ.get(layerIndex)[nodeIndex];
//                δ *= output * (1 - output);//FIXME
                list_δ.get(layerIndex)[nodeIndex] = δ;
                for (int preLayerNodeIndex = 0; preLayerNodeIndex < layerConfig[preLayerIndex]; preLayerNodeIndex++) {
                    double preLayerNodeOutput = list_σ.get(preLayerIndex)[preLayerNodeIndex];
                    list_w.get(layerIndex).get(nodeIndex)[preLayerNodeIndex] -= leanrgingRate * δ * preLayerNodeOutput;
                    list_b.get(layerIndex)[nodeIndex] -= leanrgingRate * δ;
//                    if (list_w.get(layerIndex).get(nodeIndex)[preLayerNodeIndex] > WEIGHT_MAX)
//                        list_w.get(layerIndex).get(nodeIndex)[preLayerNodeIndex] = WEIGHT_MAX;
//                    if (list_w.get(layerIndex).get(nodeIndex)[preLayerNodeIndex] < -WEIGHT_MAX)
//                        list_w.get(layerIndex).get(nodeIndex)[preLayerNodeIndex] = -WEIGHT_MAX;
                    if (list_b.get(layerIndex)[nodeIndex] > layerConfig[preLayerIndex])
                        list_b.get(layerIndex)[nodeIndex] = layerConfig[preLayerIndex];
                    if (list_b.get(layerIndex)[nodeIndex] < -layerConfig[preLayerIndex])
                        list_b.get(layerIndex)[nodeIndex] = -layerConfig[preLayerIndex];
                }
            }
        }

    }


    public void computeCost(double[] outputLabels) {
        int lastLayerIndex = layerNum - 1;
        cost = 0;
        for (int i = 0; i < outputLabels.length; i++) {//遍历最后一层节点（输出层）平方差求和 数字0用[1,0,0,0,0,0,0,0,0]表示 平均方差
            double detal = outputLabels[i] - list_σ.get(lastLayerIndex)[i];
            cost += detal * detal / 2;
        }
    }


    public String getStructureStr() {
        DecimalFormat nf = new DecimalFormat("0.0000");
        StringBuilder sb = new StringBuilder();
        sb.append("cost:").append(cost).append('\n');
        sb.append("layer0:inputs");
        for (int i = 0; i < layerConfig[0]; i++) {
            sb.append(nf.format(list_σ.get(0)[i]));
            sb.append(' ');
        }
        sb.append(']').append('\n');
        for (int i = 1; i < list_w.size(); i++) {
            ArrayList<double[]> layerWeights = list_w.get(i);
            sb.append("layer").append(i + 1).append(":\n");
            for (int j = 0; j < layerWeights.size(); j++) {
                double[] nodeWeights = layerWeights.get(j);
                sb.append("-------node:").append(j)
                        .append(" o=").append(nf.format(list_σ.get(i)[j]))
                        .append(" δ=").append(nf.format(list_δ.get(i)[j]))
                        .append(" b=").append(nf.format(list_b.get(i)[j]))
                        .append(" w=").append(ArrayUtil.toString(nodeWeights, "0.00")).append('\n');
            }
        }
        return sb.toString();
    }

    public double[] getOutputLayer() {
        return list_σ.get(layerNum - 1);
    }

    public double getLoss() {
        return cost;
    }
}
