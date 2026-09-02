import SwiftUI

struct BottomTabBar: View {
    @Binding var selectedTab: AppTab

    var body: some View {
        HStack(spacing: 0) {
            item(tab: .home, title: "Home", icon: "house")
            item(tab: .history, title: "History", icon: "clock.arrow.circlepath")
            item(tab: .profile, title: "Profile", icon: "person")
        }
        .frame(height: 49)
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
                Image(systemName: resolvedIcon(for: icon, isSelected: selectedTab == tab))
                    .font(.system(size: 19, weight: .medium))
                    .frame(height: 21)

                Text(title)
                    .font(.system(size: 10, weight: .medium))
            }
            .foregroundStyle(selectedTab == tab ? Color.slopePrimary : Color.secondary)
            .frame(maxWidth: .infinity)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selectedTab == tab ? .isSelected : [])
    }

    private func resolvedIcon(for icon: String, isSelected: Bool) -> String {
        guard isSelected, icon != "clock.arrow.circlepath" else { return icon }
        return "\(icon).fill"
    }
}
