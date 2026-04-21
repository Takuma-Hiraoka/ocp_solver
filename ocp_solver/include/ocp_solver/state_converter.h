#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>
#include "ocp_solver/contact_candidate.h"

namespace ocp_solver {
  template <typename SCALAR_T>
    class StateConverter {
  public:
  StateConverter(size_t joint_dim, const std::vector<ContactCandidate>& contactCandidates, std::unordered_map<std::string, size_t> joint_index_map, size_t base_dim=6)
    : joint_dim(joint_dim),
      contact_num(contactCandidates.size()),
      base_dim(base_dim),
      state_dim(2 * (base_dim + joint_dim)),
      input_dim(6 * contact_num + joint_dim),
      joint_index_map(joint_index_map),
      contactCandidates(contactCandidates){};
    ~StateConverter() = default;
    StateConverter* clone() const { return new StateConverter(*this); }

    size_t getStateDim() const { return state_dim; };
    size_t getInputDim() const { return input_dim; };
    size_t getBaseDim() const { return base_dim; };
    size_t getJointDim() const { return joint_dim; };
    size_t getContactNum() const { return contact_num; };
    size_t getGenCoordinatesDim() const { return base_dim + joint_dim; };
    std::vector<ContactCandidate> getContactCandidates() const { return contactCandidates; }

    size_t getBaseStartindex() const { return 0; };
    size_t getJointStartindex() const { return base_dim; };
    size_t getJointVelocitiesStartindex() const { return (2*base_dim + joint_dim); };
    size_t getJointAccelerationsStartindex() const { return (6 * contact_num); };

    size_t getContactWrenchStartIndices(size_t contactIndex) const { return 6 * contactIndex; };
    size_t getContactForceStartIndices(size_t contactIndex) const { return getContactWrenchStartIndices(contactIndex); };
    size_t getContactMomentStartIndices(size_t contactIndex) const { return getContactWrenchStartIndices(contactIndex) + 3; };

    Eigen::Matrix<SCALAR_T, -1, 1> getGeneralizedCoordinates(const Eigen::Matrix<SCALAR_T, -1, 1>& state) const {
      assert(state.size() == this->state_dim);
      return state.head((base_dim + joint_dim));
    };

    Eigen::Matrix<SCALAR_T, 6, 1> getBasePose(const Eigen::Matrix<SCALAR_T, -1, 1>& state) const {
      assert(this->base_dim == 6);
      assert(state.size() == this->state_dim);
      return state.head(6);
    };

    Eigen::Matrix<SCALAR_T, 3, 1> getBasePosition(const Eigen::Matrix<SCALAR_T, -1, 1>& state) const {
      assert(this->base_dim == 6);
      assert(state.size() == this->state_dim);
      return state.head(3);
    }

    Eigen::Matrix<SCALAR_T, 3, 1> getBaseOrientationEulerZYX(const Eigen::Matrix<SCALAR_T, -1, 1>& state) const {
      assert(this->base_dim == 6);
      assert(state.size() == this->state_dim);
      return state.segment(3, 3);
    }

    Eigen::Matrix<SCALAR_T, 3, 1> getBaseComLinearVelocity(const Eigen::Matrix<SCALAR_T, -1, 1>& state) const {
      assert(this->base_dim == 6);
      assert(state.size() == this->state_dim);
      return state.segment((base_dim + joint_dim), 3);
    }

    Eigen::Matrix<SCALAR_T, 3, 1> getBaseLinearVelocity(const Eigen::Matrix<SCALAR_T, -1, 1>& state) const {
      assert(this->base_dim == 6);
      assert(state.size() == this->state_dim);
      return state.segment((base_dim + joint_dim), 3);
    }

    // Contains Euler angle derivatives, not angular velocity!
    Eigen::Matrix<SCALAR_T, 6, 1> getBaseComVelocity(const Eigen::Matrix<SCALAR_T, -1, 1>& state) const {
      assert(this->base_dim == 6);
      assert(state.size() == this->state_dim);
      return state.segment((base_dim + joint_dim), 6);
    };

    Eigen::Matrix<SCALAR_T, 3, 1> getBaseEulerZYXDerivatives(const Eigen::Matrix<SCALAR_T, -1, 1>& state) const {
      assert(this->base_dim == 6);
      assert(state.size() == this->state_dim);
      return state.segment((base_dim + joint_dim) + 3, 3);
    }

    Eigen::Matrix<SCALAR_T, -1, 1> getJointAngles(const Eigen::Matrix<SCALAR_T, -1, 1>& state) const {
      assert(state.size() == this->state_dim);
      return state.segment(getJointStartindex(), joint_dim);
    };

    Eigen::Matrix<SCALAR_T, -1, 1> getJointVelocities(const Eigen::Matrix<SCALAR_T, -1, 1>& state, const Eigen::Matrix<SCALAR_T, -1, 1>& input) const {
      assert(state.size() == this->state_dim);
      assert(input.size() == this->input_dim);
      return state.tail(joint_dim);
    };

    Eigen::Matrix<SCALAR_T, -1, 1> getGeneralizedVelocities(const Eigen::Matrix<SCALAR_T, -1, 1>& state, const Eigen::Matrix<SCALAR_T, -1, 1>& input) {
      assert(state.size() == this->state_dim);
      return state.tail((base_dim + joint_dim));
    };

    Eigen::Matrix<SCALAR_T, -1, 1> getJointAccelerations(const Eigen::Matrix<SCALAR_T, -1, 1>& input) const {
      assert(input.size() == this->input_dim);
      return input.tail(joint_dim);
    };

    void setGeneralizedCoordinates(Eigen::Matrix<SCALAR_T, -1, 1>& state, const Eigen::Matrix<SCALAR_T, -1, 1>& generalizedCorrdinates) const {
      assert(state.size() == this->state_dim);
      assert(generalizedCorrdinates.size() == (base_dim + joint_dim));
      state.head((base_dim + joint_dim)) = generalizedCorrdinates;
    }

    void setBasePose(Eigen::Matrix<SCALAR_T, -1, 1>& state, const Eigen::Matrix<SCALAR_T, 6, 1>& basePose) const {
      assert(this->base_dim == 6);
      assert(state.size() == this->state_dim);
      state.head(6) = basePose;
    }

    void setBasePosition(Eigen::Matrix<SCALAR_T, -1, 1>& state, const Eigen::Matrix<SCALAR_T, 3, 1>& position) const {
      assert(this->base_dim == 6);
      assert(state.size() == this->state_dim);
      state.head(3) = position;
    }

    void setBaseOrientationEulerZYX(Eigen::Matrix<SCALAR_T, -1, 1>& state, const Eigen::Matrix<SCALAR_T, 3, 1>& eulerAnglesZYX) const {
      assert(this->base_dim == 6);
      assert(state.size() == this->state_dim);
      state.segment(3, 3) = eulerAnglesZYX;
    }

    void setBaseLinearVelocity(Eigen::Matrix<SCALAR_T, -1, 1>& state, const Eigen::Matrix<SCALAR_T, 3, 1>& velocity) const {
      assert(this->base_dim == 6);
      assert(state.size() == this->state_dim);
      state.segment((base_dim + joint_dim), 3) = velocity;
    }

    void setBaseOrientationEulerZYXDerivatives(Eigen::Matrix<SCALAR_T, -1, 1>& state, const Eigen::Matrix<SCALAR_T, 3, 1>& eulerAnglesZYXDerivative) const {
      assert(this->base_dim == 6);
      assert(state.size() == this->state_dim);
      state.segment((base_dim + joint_dim) + 3, 3) = eulerAnglesZYXDerivative;
    }

    void setJointAngles(Eigen::Matrix<SCALAR_T, -1, 1>& state, const Eigen::Matrix<SCALAR_T, -1, 1>& jointAngles) const {
      assert(state.size() == this->state_dim);
      state.segment(getJointStartindex(), joint_dim) = jointAngles;
    }

    void setJointVelocities(Eigen::Matrix<SCALAR_T, -1, 1>& state, Eigen::Matrix<SCALAR_T, -1, 1>& input, const Eigen::Matrix<SCALAR_T, -1, 1>& jointVelocities) const {
      assert(state.size() == this->state_dim);
      assert(input.size() == this->input_dim);
      state.tail(joint_dim) = jointVelocities;
    }

    void adaptBasePoseHeight(Eigen::Matrix<SCALAR_T, -1, 1>& state, ocs2::scalar_t heightChange) const {
      assert(this->base_dim == 6);
      assert(state.size() == this->state_dim);
      state[2] += heightChange;
    }

    Eigen::Matrix<SCALAR_T, 6, 1> getContactWrench(const Eigen::Matrix<SCALAR_T, -1, 1>& input, size_t contactIndex) const {
      assert(input.size() == this->input_dim);
      return input.segment(getContactWrenchStartIndices(contactIndex), 6);
    };

    Eigen::Matrix<SCALAR_T, 3, 1> getContactForce(const Eigen::Matrix<SCALAR_T, -1, 1>& input, size_t contactIndex) const {
      assert(input.size() == this->input_dim);
      return input.segment(getContactForceStartIndices(contactIndex), 3);
    };

    Eigen::Matrix<SCALAR_T, 3, 1> getContactMoment(const Eigen::Matrix<SCALAR_T, -1, 1>& input, size_t contactIndex) const {
      assert(input.size() == this->input_dim);
      return input.segment(getContactMomentStartIndices(contactIndex), 3);
    };

    void setContactWrench(Eigen::Matrix<SCALAR_T, -1, 1>& input, const Eigen::Matrix<SCALAR_T, 6, 1>& wrench, size_t contactIndex) const {
      assert(input.size() == this->input_dim);
      input.segment(getContactWrenchStartIndices(contactIndex), 6) = wrench;
    };

    void setContactForce(Eigen::Matrix<SCALAR_T, -1, 1>& input, const Eigen::Matrix<SCALAR_T, 3, 1>& force, size_t contactIndex) const {
      assert(input.size() == this->input_dim);
      input.segment(getContactForceStartIndices(contactIndex), 3) = force;
    };

    void setContactMoment(Eigen::Matrix<SCALAR_T, -1, 1>& input, const Eigen::Matrix<SCALAR_T, 3, 1>& moment, size_t contactIndex) const {
      assert(input.size() == this->input_dim);
      input.segment(getContactMomentStartIndices(contactIndex), 3) = moment;
    };

    size_t getJointIndex(const std::string& jointName) const {
      auto it = joint_index_map.find(jointName);
      if (it != joint_index_map.end()) {
        return it->second;  // Return the found index
      } else {
        throw std::runtime_error("Joint name " + jointName + " is not contained in MPC model!");
      }
    }

  private:
  StateConverter(const StateConverter& rhs)
    : joint_dim(rhs.joint_dim),
      contact_num(rhs.contact_num),
      state_dim(rhs.state_dim),
      input_dim(rhs.input_dim),
      base_dim(rhs.base_dim),
      joint_index_map(rhs.joint_index_map){};
  public:
    const size_t joint_dim;
    const size_t contact_num;
    const size_t state_dim;
    const size_t input_dim;
    const size_t base_dim;
    const std::unordered_map<std::string, size_t> joint_index_map;
    std::vector<ContactCandidate> contactCandidates;
  };

}
