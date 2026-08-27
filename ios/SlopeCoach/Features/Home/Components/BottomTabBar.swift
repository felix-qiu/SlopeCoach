import SwiftUI

struct BottomTabBar: View {
    @Binding var selectedTab: AppTab

    var body: some View {
        HStack(spacing: 0) {
            item(tab: .home, title: "Home", icon: "house.fill")
            item(tab: .history, title: "History", icon: "clock.arrow.circlepath")
            item(tab: .profile, title: "Profile", icon: "person.fill")
        }
        .padding(.top, 10)
        .padding(.bottom, 6)
        .background(.ultraThinMaterial)
        .overlay(alignment: .top) {
            Divider().opacity(0.55)
        }
    }

    private func item(tab: AppTab, title: String, icon: String) -> some View {
        Button {
            selectedTab = tab
        } label: {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 18, weight: .semibold))
                    .frame(height: 22)

                Text(title)
                    .font(.system(size: 10, weight: selectedTab == tab ? .semibold : .medium))
            }
            .foregroundStyle(selectedTab == tab ? Color.slopePrimary : Color.secondary)
            .frame(maxWidth: .infinity)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selectedTab == tab ? .isSelected : [])
    }
}
