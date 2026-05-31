#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include "ocp_solver/contact_candidate.h"

namespace ocp_solver {
  template <typename SCALAR_T>
    class StateConverter {
  public:
    StateConverter(size_t joint_dim, const std::vector<ContactCandidateInfoTpl<SCALAR_T>>& contactCandidates, std::unordered_map<std::string, size_t> joint_index_map, size_t base_q_dim=7, size_t base_v_dim=6)
    : joint_dim(joint_dim),
      contact_num(contactCandidates.size()),
      contact_point_search_num(countContactPointSearches(contactCandidates)),
      base_q_dim(base_q_dim),
      base_v_dim(base_v_dim),
      state_dim(base_q_dim + base_v_dim + joint_dim*2 + 3 * contact_point_search_num),
      input_dim(6 * contact_num + joint_dim + 3 * contact_point_search_num),
      joint_index_map(joint_index_map),
      contactCandidates(contactCandidates){};
    ~StateConverter() = default;
    StateConverter* clone() const { return new StateConverter(*this); }

    size_t getStateDim() const { return state_dim; };
    size_t getInputDim() const { return input_dim; };
    size_t getTangentDim() const { return base_v_dim + joint_dim; };
    size_t getStateVariableDim() const { return getStateVariableDimWithoutContactPointVariables() + 3 * contact_point_search_num; };
    size_t getStateDimWithoutContactPointVariables() const { return base_q_dim + base_v_dim + joint_dim*2; };
    size_t getStateVariableDimWithoutContactPointVariables() const { return 2*(base_v_dim + joint_dim); };
    size_t getBaseQDim() const { return base_q_dim; };
    size_t getBaseVDim() const { return base_v_dim; };
    size_t getJointDim() const { return joint_dim; };
    size_t getContactNum() const { return contact_num; };
    size_t getContactPointSearchNum() const { return contact_point_search_num; };
    size_t getGenCoordinatesDim() const { return base_q_dim + joint_dim; };
    std::vector<ContactCandidateIndex> getContactCandidateIds() const {
      std::vector<ContactCandidateIndex> ids;
      ids.reserve(contactCandidates.size());
      for (const auto& candidate : contactCandidates) ids.push_back(candidate.index);
      return ids;
    }
    const ContactCandidateInfoTpl<SCALAR_T>& getContactCandidate(size_t contactIndex) const { return contactCandidates.at(contactIndex); }
    ContactCandidateInfoTpl<SCALAR_T> getContactCandidate(const Eigen::Matrix<SCALAR_T, -1, 1>& state, size_t contactIndex) const {
      ContactCandidateInfoTpl<SCALAR_T> candidate = contactCandidates.at(contactIndex);
      if (candidate.searchContactPoint) {
        pinocchio::SE3Tpl<SCALAR_T> localPoseInLocalFrame = candidate.localPoseInLocalFrame;
        localPoseInLocalFrame.translation() = getContactPointLocalPosition(state, contactIndex);
        candidate.localPoseInLocalFrame = localPoseInLocalFrame;
        candidate.localPose = candidate.localFramePose * localPoseInLocalFrame;
      }
      return candidate;
    }

    size_t getBaseStartindex() const { return 0; };
    size_t getJointStartindex() const { return base_q_dim; };
    size_t getGeneralizedVelocitiesStartindex() const { return getGenCoordinatesDim(); };
    size_t getJointVelocitiesStartindex() const { return (base_q_dim + base_v_dim + joint_dim); };
    size_t getJointAccelerationsStartindex() const { return (6 * contact_num); };
    size_t getContactPointLocalPositionStartIndex(size_t contactIndex) const {
      const size_t searchIndex = getContactPointSearchIndex(contactIndex);
      return getStateDimWithoutContactPointVariables() + 3 * searchIndex;
    }
    size_t getContactPointLocalPositionVariableStartIndex(size_t contactIndex) const {
      const size_t searchIndex = getContactPointSearchIndex(contactIndex);
      return getStateVariableDimWithoutContactPointVariables() + 3 * searchIndex;
    }
    size_t getContactPointLocalVelocityStartIndex(size_t contactIndex) const {
      const size_t searchIndex = getContactPointSearchIndex(contactIndex);
      return 6 * contact_num + joint_dim + 3 * searchIndex;
    }

    size_t getContactWrenchStartIndices(size_t contactIndex) const { return 6 * contactIndex; };
    size_t getContactForceStartIndices(size_t contactIndex) const { return getContactWrenchStartIndices(contactIndex); };
    size_t getContactMomentStartIndices(size_t contactIndex) const { return getContactWrenchStartIndices(contactIndex) + 3; };

    Eigen::Matrix<SCALAR_T, -1, 1> getGeneralizedCoordinates(const Eigen::Matrix<SCALAR_T, -1, 1>& state) const {
      assert(state.size() >= this->state_dim);
      return state.head((base_q_dim + joint_dim));
    };

    Eigen::Matrix<SCALAR_T, -1, 1> getBasePose(const Eigen::Matrix<SCALAR_T, -1, 1>& state) const {
      assert(state.size() >= this->state_dim);
      return state.head(this->base_q_dim);
    };

    Eigen::Matrix<SCALAR_T, 3, 1> getBasePosition(const Eigen::Matrix<SCALAR_T, -1, 1>& state) const {
      assert(state.size() >= this->state_dim);
      return state.head(3);
    }

    Eigen::Matrix<SCALAR_T, 3, 1> getBaseLinearVelocity(const Eigen::Matrix<SCALAR_T, -1, 1>& state) const {
      assert(state.size() >= this->state_dim);
      return state.segment((base_q_dim + joint_dim), 3);
    }

    Eigen::Matrix<SCALAR_T, -1, 1> getJointAngles(const Eigen::Matrix<SCALAR_T, -1, 1>& state) const {
      assert(state.size() >= this->state_dim);
      return state.segment(getJointStartindex(), joint_dim);
    };

    Eigen::Matrix<SCALAR_T, -1, 1> getJointVelocities(const Eigen::Matrix<SCALAR_T, -1, 1>& state, const Eigen::Matrix<SCALAR_T, -1, 1>& input) const {
      assert(state.size() >= this->state_dim);
      assert(input.size() >= this->input_dim);
      return state.segment(getJointVelocitiesStartindex(), joint_dim);
    };

    Eigen::Matrix<SCALAR_T, -1, 1> getGeneralizedVelocities(const Eigen::Matrix<SCALAR_T, -1, 1>& state, const Eigen::Matrix<SCALAR_T, -1, 1>& input) const {
      assert(state.size() >= this->state_dim);
      return state.segment(getGeneralizedVelocitiesStartindex(), getTangentDim());
    };

    Eigen::Matrix<SCALAR_T, -1, 1> getJointAccelerations(const Eigen::Matrix<SCALAR_T, -1, 1>& input) const {
      assert(input.size() >= this->input_dim);
      return input.segment(getJointAccelerationsStartindex(), joint_dim);
    };
    Eigen::Matrix<SCALAR_T, 3, 1> getContactPointLocalPosition(const Eigen::Matrix<SCALAR_T, -1, 1>& state, size_t contactIndex) const {
      assert(state.size() >= this->state_dim);
      if (!contactCandidates.at(contactIndex).searchContactPoint) {
        return contactCandidates.at(contactIndex).localPoseInLocalFrame.translation();
      }
      return state.template segment<3>(getContactPointLocalPositionStartIndex(contactIndex));
    }
    Eigen::Matrix<SCALAR_T, 3, 1> getDefaultContactPointLocalPosition(size_t contactIndex) const {
      return contactCandidates.at(contactIndex).localPoseInLocalFrame.translation();
    }
    Eigen::Matrix<SCALAR_T, 3, 1> getContactPointLocalVelocity(const Eigen::Matrix<SCALAR_T, -1, 1>& input, size_t contactIndex) const {
      assert(input.size() >= this->input_dim);
      if (!contactCandidates.at(contactIndex).searchContactPoint) {
        return Eigen::Matrix<SCALAR_T, 3, 1>::Zero();
      }
      return input.template segment<3>(getContactPointLocalVelocityStartIndex(contactIndex));
    }

    void setJointAngles(Eigen::Matrix<SCALAR_T, -1, 1>& state, const Eigen::Matrix<SCALAR_T, -1, 1>& jointAngles) const {
      assert(state.size() >= this->state_dim);
      state.segment(getJointStartindex(), joint_dim) = jointAngles;
    }

    void setJointVelocities(Eigen::Matrix<SCALAR_T, -1, 1>& state, Eigen::Matrix<SCALAR_T, -1, 1>& input, const Eigen::Matrix<SCALAR_T, -1, 1>& jointVelocities) const {
      assert(state.size() >= this->state_dim);
      assert(input.size() >= this->input_dim);
      state.segment(getJointVelocitiesStartindex(), joint_dim) = jointVelocities;
    }

    Eigen::Matrix<SCALAR_T, 6, 1> getContactWrench(const Eigen::Matrix<SCALAR_T, -1, 1>& input, size_t contactIndex) const {
      assert(input.size() >= this->input_dim);
      return input.segment(getContactWrenchStartIndices(contactIndex), 6);
    };

    Eigen::Matrix<SCALAR_T, 3, 1> getContactForce(const Eigen::Matrix<SCALAR_T, -1, 1>& input, size_t contactIndex) const {
      assert(input.size() >= this->input_dim);
      return input.segment(getContactForceStartIndices(contactIndex), 3);
    };

    Eigen::Matrix<SCALAR_T, 3, 1> getContactMoment(const Eigen::Matrix<SCALAR_T, -1, 1>& input, size_t contactIndex) const {
      assert(input.size() >= this->input_dim);
      return input.segment(getContactMomentStartIndices(contactIndex), 3);
    };

    void setContactWrench(Eigen::Matrix<SCALAR_T, -1, 1>& input, const Eigen::Matrix<SCALAR_T, 6, 1>& wrench, size_t contactIndex) const {
      assert(input.size() >= this->input_dim);
      input.segment(getContactWrenchStartIndices(contactIndex), 6) = wrench;
    };

    void setContactForce(Eigen::Matrix<SCALAR_T, -1, 1>& input, const Eigen::Matrix<SCALAR_T, 3, 1>& force, size_t contactIndex) const {
      assert(input.size() >= this->input_dim);
      input.segment(getContactForceStartIndices(contactIndex), 3) = force;
    };

    void setContactMoment(Eigen::Matrix<SCALAR_T, -1, 1>& input, const Eigen::Matrix<SCALAR_T, 3, 1>& moment, size_t contactIndex) const {
      assert(input.size() >= this->input_dim);
      input.segment(getContactMomentStartIndices(contactIndex), 3) = moment;
    };

    size_t getJointIndex(const std::string& jointName) const {
      typename std::unordered_map<std::string, size_t>::const_iterator it = joint_index_map.find(jointName);
      if (it != joint_index_map.end()) {
        return it->second;  // Return the found index
      } else {
        throw std::runtime_error("Joint name " + jointName + " is not contained in MPC model!");
      }
    }

  private:
  static size_t countContactPointSearches(const std::vector<ContactCandidateInfoTpl<SCALAR_T>>& candidates) {
    size_t count = 0;
    for (const auto& candidate : candidates) {
      if (candidate.searchContactPoint) ++count;
    }
    return count;
  }
  size_t getContactPointSearchIndex(size_t contactIndex) const {
    const auto& candidate = contactCandidates.at(contactIndex);
    if (!candidate.searchContactPoint) {
      throw std::runtime_error("Contact candidate " + candidate.frameName + " does not search a contact point.");
    }
    return candidate.contactPointStateIndex;
  }
  StateConverter(const StateConverter& rhs)
    : joint_dim(rhs.joint_dim),
      contact_num(rhs.contact_num),
      contact_point_search_num(rhs.contact_point_search_num),
      state_dim(rhs.state_dim),
      input_dim(rhs.input_dim),
      base_q_dim(rhs.base_q_dim),
      base_v_dim(rhs.base_v_dim),
      joint_index_map(rhs.joint_index_map),
      contactCandidates(rhs.contactCandidates){};
  public:
    const size_t joint_dim;
    const size_t contact_num;
    const size_t contact_point_search_num;
    const size_t state_dim;
    const size_t input_dim;
    const size_t base_q_dim;
    const size_t base_v_dim;
    const std::unordered_map<std::string, size_t> joint_index_map;
    std::vector<ContactCandidateInfoTpl<SCALAR_T>> contactCandidates; // state等のwrenchの順番はcontactCandidatesに登録された順
  };

}
