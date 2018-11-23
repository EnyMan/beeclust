class State{
  public:
    int x;
    int y;
    int d;
    int t;

    State() {}

    State(int x, int y, int d, int t){
      this->x = x;
      this->y = y;
      this->d = d;
      this->t = t;
    }

    bool operator<(const State& rhs) const{
      return this->x < rhs.x || ((!(rhs.x < this->x)) && (this->y < rhs.y));
    }
};
