#include <iostream>
#include <math.h>

using namespace std;

void print_data(double x[], int data_set_size){

    for(int i = 0; i < data_set_size; i++){

        cout << x[i] << "   ";
    }

    cout << "\n\n\n" << endl;
}

void compute_model(double x[], double y_pred[], int data_size, double w, double b){

    for(int i = 0; i < data_size; i++){

        y_pred[i] = ( w * x[i] ) + b;
    }

}

double compute_cost(double x[], double y[], int data_size, double w, double b){

    double total_cost = 0;

    for (int i = 0; i < data_size; i++){
        
        double fwb = ( w * x[i] ) + b;
        fwb = fwb - y[i];

        total_cost += fwb;
    }

    total_cost = pow(total_cost, 2);

    total_cost /= ( 2 * data_size);

    return total_cost;
}

pair<double, double>  compute_gradient(double x[], double y[], int data_Size, double w, double b){
    
    double dw = 0;
    double db = 0;

    for (int i = 0; i < data_Size; i++){

        double fwb = ( w * x[i] ) + b;
        double dw_i = ( fwb - y[i] ) * x[i];
        double db_i = ( fwb - y[i] );

        dw += dw_i;
        db += db_i;
    }

    dw /= data_Size;
    db /= data_Size;

    return make_pair(dw, db);
}

pair<double, double> compute_gradient_descent(double x[], double y[], int data_size, double w, double b, double alpha){

    pair<double, double> dw_db = compute_gradient(x, y, data_size, w, b);

    w = w - ( alpha * dw_db.first );
    b = b - ( alpha * dw_db.second );

    return make_pair(w, b);
}

pair<double, double> train_model(double x[], double y[], double y_pred[], int data_size, double w, double b, double alpha, int epochs){

    for (int i = 0; i < epochs; i++){

        compute_model(x, y_pred, data_size, w, b);
        double cost = compute_cost(x, y, data_size, w, b);
        pair<double, double> w_b = compute_gradient_descent(x, y, data_size, w, b, alpha);

        w = w_b.first;
        b = w_b.second;

        cout << "--- COST = " << cost << " --- w = " << w << " --- b = " << b << " --- Remaining = " << epochs - ( i + 1) << endl;
    }

    return make_pair(w, b);
}

int main()
{

    double x_train[10] = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45};

    double y_train[10] = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};

    double y_pred[10];

    int data_size = size(x_train);

    double w = 0, b = 0, alpha = 0.001;

    int epochs = 100000;

    pair<double, double> w_b = train_model(x_train, y_train, y_pred, data_size, w, b, alpha, epochs);

    // make predictions

    cout << "\n\n\n" << endl;

    print_data(y_pred, data_size);

    return 0;
}