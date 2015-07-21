#include "TwoPassArm.h"
void tp::mergeSets(tp::lbSet *u, tp::lbSet *v, tp::lbSet *u_tail, std::vector<lbSet*>& table)
{
    //hang v to the end of u
    u_tail->next = v;
    u->tail=v->tail;
    //update all related sets in the table;
    while(v->label != -1){
        //index is the name of the set
        int index= v->label-1;
        //value is the representitive, which in our case is u
        table.at(index)=u;
        v=v->next;
    }
}

void tp::resolve(const int& x,const int& y,std::vector<lbSet*>& table){
    tp::lbSet *u=table.at(x-1);
    tp::lbSet *v=table.at(y-1);
    if((u->label)>(v->label))
    {
        tp::mergeSets(u,v,u->tail,table);
    }else if((u->label)<(v->label)){
        tp::mergeSets(v,u,v->tail,table);
    }
}

void tp::detectRun(Mat& source, int row, std::vector<Point>& curRuns)
{
    int startPos=0;
    for(int j=0;j<source.cols-1;j++)
    {
        if(source.at<int>(row,j)==1)
        {
            if(startPos==0)
            {
                startPos=j;
            }
        }else{
            if(startPos!=0)
            {
                curRuns.push_back(Point(startPos,j-1));
                startPos=0;
            }
        }
    }
    //last pixel of the row should be considered separatly
    if(source.at<int>(row,source.cols-1)==1)
    {
        if(startPos!=0)
        {
            curRuns.push_back(Point(startPos,source.cols-1));
        }else{
            curRuns.push_back(Point(source.cols-1,source.cols-1));
        }
    }else{
        if(startPos!=0)
        {
            curRuns.push_back(Point(startPos,source.cols-2));
        }
    }

}

std::vector<int> tp::calc2Pass(Mat& source)
{
    //first pass
    std::vector<int> res;
    std::vector<lbSet*> table;
    std::vector<Point> preRuns;
    std::vector<Point> curRuns;
    std::vector<Point> temp;
    int label=1;
    tp::lbSet *minusOne= new lbSet;
    minusOne->label=-1;
    for(int i=0;i<source.rows;i++)
    {
        //swap the runs
        temp=preRuns;
        preRuns=curRuns;
        curRuns=temp;
        curRuns.clear();
        tp::detectRun(source,i,curRuns);
        for(int m=0;m<curRuns.size();m++)
        {
            int preLabel = 0;
            Point curRun=curRuns.at(m);
            //try to find neighbours
            for(int n=0;n<preRuns.size();n++)
            {
                Point preRun=preRuns.at(n);
                if((curRun.x>=preRun.x-1&&curRun.x<=preRun.y+1)||(curRun.y>=preRun.x-1&&curRun.y<=preRun.y+1)||(curRun.x<=preRun.x&&curRun.y>=preRun.y))
                {

                    if(preLabel==0){
                    //if it is the first neighbour label it with correspounding number
                        preLabel=source.at<int>(i-1,preRun.x);
                        for(int k=curRun.x;k<curRun.y+1;k++)
                        {
                            source.at<int>(i,k)=preLabel;
                        }
                    }else{
                    //otherwise merge the sets
                        tp::resolve(source.at<int>(i-1,preRun.x),preLabel,table);
                    }
                }

            }
            if(preLabel==0)
            {
                //if no neighbours are found, create new set with a new label number
                tp::lbSet *newSet= new lbSet;
                newSet->label=label;
                newSet->tail=newSet;
                newSet->next=minusOne;
                table.push_back(newSet);
                for(int k=curRun.x;k<curRun.y+1;k++)
                {
                    source.at<int>(i,k)=label;
                }
                label++;

            }
        }
    }


    //second pass
    int curlabel=0;
    for(int i=0;i<source.rows;i++)
    {
        for(int j=0;j<source.cols;j++)
        {
            if(source.at<int>(i,j)!=0){
                int index=source.at<int>(i,j)-1;
                source.at<int>(i,j)=table.at(index)->label;
            }
        }
    }


    return res;
}
