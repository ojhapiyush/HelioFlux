#include <bits/stdc++.h>
using namespace std;

struct Interval {
    int start;
    double cost;
};

bool compare(Interval a, Interval b) {
    return a.cost < b.cost;
}

void readCSV(const string& filename, vector<int>& ratings, vector<int>& hours, vector<string>& types) {
    ifstream file(filename);
    string line, value;
    getline(file, line);
    while (getline(file, line)) {
        stringstream ss(line);
        string rating, hour, type;
        
        getline(ss, rating, ',');
        getline(ss, hour, ',');
        getline(ss, type, ',');
        
        ratings.push_back(stoi(rating));
        hours.push_back(stoi(hour));
        types.push_back(type);
    }
    
    file.close();
}

vector<double> readPricesFromFile(const string& fileName) {
    vector<double> prices;
    ifstream file(fileName);

    if (!file.is_open()) {
        cerr << "Error: Could not open the file." << endl;
        return prices;
    }

    string line;
    bool firstLine = true;
    while (getline(file, line)) {
        if (firstLine) {
            if (line.size() >= 3 && (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF) {
                line = line.substr(3);
            }
            firstLine = false;
        }
        try {
            double price = std::stod(line);
            prices.push_back(price);
        } catch (const std::invalid_argument& e) {
            cerr << "Invalid data in the file: " << line << endl;
        }
    }

    file.close();
    return prices;
}

pair<vector<int>, double> findOptimalSlots(const vector<double>& prices, int load_hours, int max_interval) {
    int n = prices.size();
    vector<Interval> intervals;
    for (int i = 0; i <= n - max_interval; ++i) {
        double current_sum = 0.0;
        for (int j = i; j < i + max_interval; ++j) {
            current_sum += prices[j];
        }
        intervals.push_back({i, current_sum});
    }
    sort(intervals.begin(), intervals.end(), compare);
    vector<int> optimal_slots;
    int total_hours = 0;
    double total_cost = 0.0;
    
    for (auto interval : intervals) {
        if (total_hours + max_interval <= load_hours) {
            optimal_slots.push_back(interval.start);
            total_hours += max_interval;
            total_cost += interval.cost;
        }
        if (total_hours >= load_hours) break;
    }

    return {optimal_slots, total_cost};
}

void schedule(vector<double> &prices, int load_hours, int max_interval, double appliance_rating){
    pair<vector<int>, double> result = findOptimalSlots(prices, load_hours, max_interval);
    vector<int> optimal_slots = result.first;
    double total_cost = result.second;

    cout << "Optimal time slots to run the load: \n";
    sort(optimal_slots.begin(), optimal_slots.end());
    for (int slot : optimal_slots) {
        cout << slot+1 << ":00 to " << slot+max_interval+1 << ":00 Hours" << endl;
    }
    cout << endl;

    cout << "Total energy cost: Rs. " << appliance_rating*total_cost << endl;

}

int main() {
    vector<double> prices = readPricesFromFile("../dataset/predictions.csv");
    vector<int> Appliance_rating;
    vector<int> Load_hours;
    vector<string> Appliance_type;
    readCSV("../dataset/Appliance.csv", Appliance_rating, Load_hours, Appliance_type);
    for(int i=0; i<Appliance_rating.size(); i++){
        if(Appliance_type[i] == "Regulatable Load"){
            cout << "For " << Appliance_type[i] << " with rating " << Appliance_rating[i] << "W and " << Load_hours[i] << " hours of operation\n";
            schedule(prices, Load_hours[i], 1, Appliance_rating[i]);
            cout << endl;
        }
        else{
            cout << "For " << Appliance_type[i] << " with rating " << Appliance_rating[i] << "W and " << Load_hours[i] << " hours of operation\n";
            schedule(prices, Load_hours[i], Load_hours[i], Appliance_rating[i]);
            cout << endl;
        }
    }
    return 0;
}
