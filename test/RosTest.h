#pragma once

#include <gtest/gtest.h>

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace test {

// The yaml the launch loads for a node, so the tests run what ships.
inline rclcpp::NodeOptions shippedConfig(const std::string& file) {
  return rclcpp::NodeOptions().arguments(
      {"--ros-args", "--params-file", std::string(PEOPLE_PERCEPTION_CONFIG_DIR) + "/" + file});
}

// A node under test and a helper node (io_) that publishes its inputs and collects its outputs, spun together.
class RosTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }

  void SetUp() override {
    io_ = std::make_shared<rclcpp::Node>("test_io");
    executor_.add_node(io_);
  }

  bool spinUntil(const std::function<bool()>& done, std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!done()) {
      if (std::chrono::steady_clock::now() > deadline) {
        return false;
      }
      executor_.spin_once(std::chrono::milliseconds(10));
    }
    return true;
  }

  void spinFor(std::chrono::milliseconds duration) {
    const auto end = std::chrono::steady_clock::now() + duration;
    spinUntil([&] { return std::chrono::steady_clock::now() > end; }, duration + std::chrono::seconds(1));
  }

  // Everything published on a topic from now on. Queue 20 and default (reliable) QoS, as TARE and SC-PGO subscribe.
  template <typename Message>
  std::shared_ptr<std::vector<Message>> collect(const std::string& topic) {
    auto received = std::make_shared<std::vector<Message>>();
    subscriptions_.push_back(io_->create_subscription<Message>(
        topic, 20, [received](typename Message::ConstSharedPtr msg) { received->push_back(*msg); }));
    return received;
  }

  rclcpp::executors::SingleThreadedExecutor executor_;
  rclcpp::Node::SharedPtr io_;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subscriptions_;
};

}  // namespace test
