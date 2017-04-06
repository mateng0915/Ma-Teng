public class BP {  
     

    /****神经网络结构***/ 

    //样本输入输出in&out  
    private double[] in;  
    private double[] out;  

    //隐藏层输入输出hidden_in*hidden_out  
    private double[] hidden_in;  
    private double[] hidden_out;  

    //输出层输入与输出out_in&out_out;  
    private double[] out_in;  
    private double[] out_out; 
    
    /****神经网络结构***/ 


    //各节点之间的权值w[i-h]&v[h-o]  
    private double[][] w;
    private double[][] v;

    //隐藏层和输出层的阈值hidden_y,out_y  
    private double[] hidden_y;  
    private double[] out_y;  


    //输入层隐藏层输出层节点数inputNum&hiddenNum&outputNum  
    private int inputNum;  
    private int hiddenNum;  
    private int outputNum; 

    //隐藏层输出层的一般误差  
    private double[] delta_hidden;  
    private double[] delta_out;  
    //总误差error  
    public double error;  
    //用于执行速率  
    private double rate_w;  
    private double rate_y; 
    //误差MSE
    public double sqr_err; 

     public BP(int inputNum, int hiddenNum, int outputNum, double rate_w,  
            double rate_y) {  
        super();  
        this.inputNum = inputNum;  
        this.hiddenNum = hiddenNum;  
        this.outputNum = outputNum;  
        this.rate_w = rate_w;  
        this.rate_y = rate_y;  
        in = new double[inputNum];  
        out = new double[outputNum];  
        hidden_in = new double[hiddenNum];  
        hidden_out = new double[hiddenNum];  
        out_in = new double[outputNum];  
        out_out = new double[outputNum];  
        w = new double[inputNum][hiddenNum];  
        v = new double[hiddenNum][outputNum];  
        hidden_y = new double[hiddenNum+1];  
        out_y = new double[outputNum+1];  
        delta_hidden = new double[hiddenNum];  
        delta_out = new double[outputNum];  
        RandomWeight();  
    }  

    //随机产生权值

    private void RandomWeight() {  
        RandomWeight(inputNum,hiddenNum,w,hidden_y);  // w -> 输入层到隐藏层的权值，hidden_y -> 隐藏层的阀值（偏值）
        RandomWeight(hiddenNum,outputNum,v,out_y);  // v -> 隐藏层到输出层的权值，out_y -> 输出层的偏值
    }  
  
    private void RandomWeight(int start, int end, double[][] weight, double[] yuzhi) {  
        for(int n = 0; n < end; n++)  
        {  
            for(int m = 0; m < start; m++)  
            {  
                weight[m][n] = (Math.random()/32767.0)*2-1;  
            }  
            yuzhi[n] = (Math.random()/32767.0)*2-1;  
        }  
    }  

    // 一次的训练过程

    public void train(double[] in, double[] out)  
    {  
        this.in = in;  
        this.out = out;  
        forward();  
        Calculate_err();  
        UpData();  

    }  

    private void forward() {  
        //输入层到隐藏层
        forward(inputNum,hiddenNum,w,hidden_y,in,hidden_in,hidden_out); 
        //隐藏层到输出层 
        forward(hiddenNum,outputNum,v,out_y,hidden_out,out_in,out_out);  
        error = 0;  
        for(int k=0;k<outputNum;k++)  
        {  
         //   System.out.println("in the forward function");
         //   System.out.println("解开下一行的注释");
         //   System.out.println("real:"+out[k]+"    test:"+out_out[k]);  
        }  
          
    }  

    //完成由输入层向前传送，即完成BP算法的第一步
    private void forward(int start, int end, double[][] weight, double[] yuzhi, double[] setIn,double[] begin, double[] after) {  
        //inputNum,hiddenNum,w,hidden_y,in,hidden_in,hidden_out  
        for(int n = 0; n < end; n++)  
        {  
            double sum = 0;  
            for(int m = 0; m < start; m++)  
                sum += setIn[m] * weight[m][n];  
            begin[n] = sum - yuzhi[n];  
            after[n] = Sigmoid(begin[n]);  
        }  
          
    }  

    //对输出值进行一个非线性的转换
     private double Sigmoid(double d) {  
        // TODO Auto-generated method stub  
        return 1/(1+Math.exp(-d));  
    }  
    //计算误差
    private void Calculate_err() {  
        Calculate_err_out();  
        Calculate_err_hidden();  
          
    }  
    //计算输出层的误差
    private void Calculate_err_out() {  
        sqr_err = 0;  
        for(int k = 0; k < outputNum; k++)  
        {  
            delta_out[k] = (out[k]-out_out[k]) * out_out[k] * (1-out_out[k]);  
            sqr_err += (out[k]-out_out[k])*(out[k]-out_out[k]);  
        }  
        sqr_err = sqr_err/2;  
    }  
    //计算隐藏层的误差
    private void Calculate_err_hidden() {  
        for(int n = 0;n < hiddenNum;n++)  
        {  
            double sum = 0;  
            for(int k = 0;k < outputNum;k++)  
                sum += delta_out[k] * v[n][k];  
            delta_hidden[n] = sum * hidden_out[n] * (1-hidden_out[n]);  
        }  
    }  
    //方向更新权值和偏移值

    private void UpData() {  
        UpData_v();  
        UpData_w();  
    } 


    //update out linear Weight
    private void UpData_v() {  
        for(int k=0;k<outputNum;k++)  
        {  
            for(int n=0;n<hiddenNum;n++) {
		     
                v[n][k] = v[n][k] + rate_w * delta_out[k] * hidden_out[n];
		//System.out.println("Hidden Layer -> Out Layer W"+ n +"-"+ k +" = " + v[n][k]);  
            }
	    out_y[k] = out_y[k] + rate_w * delta_out[k];   
        } 

    } 


    //update input linear Weight
    private void UpData_w() {  
        for(int n=0;n<hiddenNum;n++)  
        {  
            for(int m=0;m<inputNum;m++){
		      
                w[m][n] = w[m][n] + rate_y * delta_hidden[n] * in[m];  
		//System.out.println("Input Layer -> Hidden Layer W"+ m +"-"+ n +" = " + w[m][n]);
            }
	    hidden_y[n] = hidden_y[n] + rate_y * delta_hidden[n]; 
        }  
    }  
      
    //训练数据开始入口
    public double[] test(double[] in)  
    {  
        this.in = in;  
        forward();  
        return out_out;  
    } 

	public double[][] getW() {
		        return w;
			    }

			        public void setW(double[][] w) {
					        this.w = w;
						    }

						        public double[][] getV() {
								        return v;
									    }

									        public void setV(double[][] v) {
											        this.v = v;
												    }

  
}  
